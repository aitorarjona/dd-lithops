import lithops
import redis
import numpy as np
import cloudpickle
import types
import logging
import io
import queue
import itertools
import hashlib
import msgpack
import threading
from functools import reduce

from .utils import extract_redis_config, clean_keys
from .collectiveops import CollectiveOPs

DEBUG = True
logger = logging.getLogger('lithops')

SENTINEL = '0'


class _TaskGroup:
    def __init__(self, worker_id, group_id, redis_client):
        self._worker_id = worker_id
        self._group_id = group_id
        self._group_size = -1
        self._redis = redis_client
        self._redis_pubsub = redis_client.pubsub()
        self._transaction_counter = itertools.count(0)

    def all_reduce(self, data, operation):
        data_pickle = cloudpickle.dumps(data)
        key = '_'.join([self._group_id, str(next(self._transaction_counter)).zfill(3), 'all-reduce'])
        index = self._redis.lpush(key, data_pickle)

        logger.debug('[{}] Performing all-reduce OP with key {}'.format(self._worker_id, key))

        if index == self._group_size:
            logger.debug('[{}] Reducing partial results of {}'.format(self._worker_id, key))
            all_data_pickle = self._redis.lrange(key, 0, index)
            all_data = [cloudpickle.loads(data_pickle) for data_pickle in all_data_pickle]
            if operation == CollectiveOPs.SUM:
                result = reduce(lambda x, y: x + y, all_data)
            else:
                raise Exception('Unknown operation {}'.format(operation))
            result_pickle = cloudpickle.dumps(result)
            self._redis.set(key + '_result', result_pickle)
            self._redis.publish(key + '_topic', msgpack.packb({'result_key': key + '_result'}))
            logger.debug('[{}] Notify results of {}'.format(self._worker_id, key))
        else:
            q = queue.Queue()

            def event_handler(raw_msg):
                if 'type' not in raw_msg or raw_msg['type'] != 'message':
                    raise Exception(raw_msg)
                msg = msgpack.unpackb(raw_msg['data'])
                logger.debug('[{}] AllReduce callback received message! {}'.format(self._worker_id, msg))
                # print(msg)
                q.put(msg)

            self._redis_pubsub.subscribe(**{key + '_topic': event_handler})
            self._redis_pubsub.run_in_thread(sleep_time=1)
            logger.debug('[{}] Waiting results of {}'.format(self._worker_id, key))
            msg = q.get()
            logger.debug('[{}] Waiting results of {}'.format(self._worker_id, key))
            result_pickle = self._redis.get(msg['result_key'])
            result = cloudpickle.loads(result_pickle)
        # print(result)
        return result


def _task_worker(id, data_partition, group_id):
    logger.debug('[{}] Worker {} of group {} start'.format(id, id, group_id))
    redis_conf = extract_redis_config()
    red = redis.Redis(**redis_conf)
    red_pubsub = red.pubsub()

    q = queue.Queue()

    task_group_proxy = _TaskGroup(worker_id=id, group_id=group_id, redis_client=red)

    logger.debug('[{}] Getting data chunk {}'.format(id, data_partition.key))
    data_chunk = data_partition.get()
    func_cache = {}

    logger.debug('[{}] Getting task log'.format(id))
    tasks_packd = red.lrange(group_id + '_tasklog', 0, -1)
    tasks = [msgpack.unpackb(task_packd) for task_packd in tasks_packd]
    logger.debug('[{}] Restored {} tasks'.format(id, len(tasks)))
    for task in tasks:
        q.put(task)

    def event_handler(raw_msg):
        if 'type' not in raw_msg or raw_msg['type'] != 'message':
            raise Exception(raw_msg)
        msg = msgpack.unpackb(raw_msg['data'])
        logger.debug('[{}] Received message! {}'.format(id, msg))
        q.put(msg)

    logger.debug('[{}] Subscribe to topic {}'.format(id, group_id))
    red_pubsub.subscribe(**{group_id + '_chan': event_handler})
    red_pubsub.run_in_thread(sleep_time=1)

    worker_loop = True
    while worker_loop:
        try:
            msg = q.get(timeout=20)
            if msg['action'] == 'task':
                task = types.SimpleNamespace(**msg)
                if task.func_key in func_cache:
                    f = func_cache[task.func_key]
                else:
                    func_pickle = red.hget(group_id, task.func_key)
                    f = cloudpickle.loads(func_pickle)
                    func_cache[task.func_key] = f
                task_group_proxy._group_size = task.group_size
                args_pickle = red.hget(group_id, task.args_key)
                args = cloudpickle.loads(args_pickle)
                logger.debug('[{}] Going to execute task {}'.format(id, task.task_id))
                # print(f, args)
                result = f(data_chunk, task_group_proxy, *args)
                result_pickle = cloudpickle.dumps(result)
                pipe = red.pipeline()
                pipe.incr(task.task_join_counter, 1).hset(task.task_id, id, result_pickle)
                cnt, _ = pipe.execute()
                if cnt == task.group_size:
                    red.lpush(task.task_join_bl, cnt)
            else:
                logger.debug('Message is {}, terminating worker'.format(msg))
                worker_loop = False
        except queue.Empty as e:
            print('empty message')
            worker_loop = False

    logger.debug('[{}] Worker {} of group {} end'.format(id, id, group_id))


class PartitionedArray:
    def __init__(self):
        raise Exception('Dont instantiate this class directly, use factory methods instead')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    @classmethod
    def from_numpy(cls, arr, partitions):
        self = cls.__new__(cls)
        self._setup()
        self._partition_numpy_shards(arr, partitions)
        return self

    def _partition_numpy_shards(self, arr, partitions):
        self._arr = arr
        self._partitions = partitions
        store = lithops.storage.Storage()
        self._shards = [_NumpyPartitionedArrayShard(store, self._group_id, i, chunk) for i, chunk in
                        enumerate(np.split(arr, partitions))]

    def _setup(self):
        self._lithops_executor = lithops.FunctionExecutor()
        self._group_id = '-'.join(['lithops', self._lithops_executor.executor_id, 'group'])
        self._red = redis.Redis(**extract_redis_config())
        self._pool_active = False
        self._task_counter = itertools.count(0)
        self._func_cache = {}
        self._worker_futures = []

    def parallel_apply(self, f, *args, flatten=None):
        task_id = self._group_id + '-' + str(next(self._task_counter)).zfill(3)

        func_pickle = cloudpickle.dumps(f)
        func_pickle_hash = hashlib.md5(func_pickle).hexdigest()
        func_key = 'func-{}'.format(func_pickle_hash)
        if func_pickle_hash not in self._func_cache:
            self._red.hset(self._group_id, func_key, func_pickle)
            self._func_cache[func_pickle_hash] = func_pickle

        args_pickle = cloudpickle.dumps(args)
        args_key = 'args-{}'.format(task_id)
        self._red.hset(self._group_id, args_key, args_pickle)

        task = {'action': 'task',
                'task_id': task_id,
                'group_size': self._partitions,
                'task_join_bl': '{}-bl'.format(task_id),
                'task_join_counter': '{}-cnt'.format(task_id),
                'func_key': func_key, 'args_key': args_key}

        msg = msgpack.packb(task)
        logger.debug('Submit task {} from host'.format(task_id))
        self._red.publish(self._group_id + '_chan', msg)
        self._red.lpush(self._group_id + '_tasklog', msg)

        if not self._pool_active:
            if not DEBUG:
                logger.debug('Using lithops')
                self._worker_futures = self._lithops_executor.map(_task_worker, self._shards,
                                                                  extra_args={'group_id': self._group_id})
            else:
                logger.debug('Using threading')
                proc_args = [(i, shard, self._group_id) for i, shard in enumerate(self._shards)]
                self._worker_futures = [threading.Thread(target=_task_worker, args=p_args) for p_args in
                                        proc_args]
                for p in self._worker_futures:
                    p.start()
            self._pool_active = True

        results = None

        wait_loop = True
        logger.debug('Host for task {} completion on list {}...'.format(task_id, task['task_join_bl']))
        while wait_loop:
            res = self._red.blpop(task['task_join_bl'], timeout=5)
            logger.debug('Time out reached, trying to get functions results...')
            if res is None:
                if not DEBUG:
                    worker_status = self._lithops_executor.get_result(fs=self._worker_futures)
                    # workers_done = all([fut.ready for fut in self._worker_futures])
                    # print(workers_done)
                    # if workers_done:
                    #     worker_status = self._lithops_executor.get_result(fs=self._worker_futures)
                else:
                    workers_done = all([p.is_alive() for p in self._worker_futures])
                    if workers_done:
                        worker_status = [p.join() for p in self._worker_futures]
            else:
                logger.debug('Task {} complete'.format(task_id))
                results_pickle = self._red.hgetall(task['task_id']).values()
                # print(results)
                results = [cloudpickle.loads(res) for res in results_pickle]
                if flatten:
                    results = flatten(results)
                wait_loop = False
        print(results)
        return results

    def terminate(self):
        # storage = lithops.storage.Storage()
        # clean_keys(self._shards, storage=storage)
        print('terminate')


class _NumpyPartitionedArrayShard:
    def __init__(self, storage, group_id, part, chunk):
        self._obj_key = '_'.join([group_id, 'part{}'.format(part), type(chunk).__name__])
        logger.debug('Upload shard {}'.format(self._obj_key))
        with io.BytesIO() as file:
            np.save(file, chunk)
            file.seek(0)
            storage.put_object(storage.bucket, self._obj_key, file)

    @property
    def key(self):
        return self._obj_key

    def get(self, storage=None):
        store = storage or lithops.storage.Storage()
        blob = store.get_object(store.bucket, self._obj_key)
        with io.BytesIO() as file:
            file.write(blob)
            file.seek(0)
            arr = np.load(file)
        return arr
