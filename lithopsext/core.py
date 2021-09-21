import redis
import cloudpickle
import types
import logging
import queue
import itertools
import msgpack
from functools import reduce

from .utils import extract_redis_config

logger = logging.getLogger('lithops')

TASK_GROUP_GLOBAL = None


def get_group():
    if TASK_GROUP_GLOBAL:
        return TASK_GROUP_GLOBAL
    else:
        raise Exception('There is no group for this task!')


class _TaskGroup:
    def __init__(self, worker_id, group_id, redis_client):
        self._worker_id = worker_id
        self._group_id = group_id
        self._group_size = -1
        self._redis = redis_client
        self._redis_pubsub = redis_client.pubsub()
        self._transaction_counter = itertools.count(0)

    def sync(self, data, reducer=None, initial_value=None, gatherer=None):
        sync_key = '_'.join([self._group_id, str(next(self._transaction_counter)).zfill(3), 'sync'])
        logger.debug('[{}] Syncing {}'.format(self._worker_id, sync_key))

        sync_result = None
        if self._worker_id == 0:
            if reducer:
                accum = reducer(data, initial_value)
                reduced = 1

                while reduced < self._group_size:
                    _, raw_value = self._redis.blpop(sync_key)
                    logger.debug('[{}] Got reduce value'.format(self._worker_id))
                    value = cloudpickle.loads(raw_value)
                    # print('Value got is', value)
                    accum = reducer(value, accum)
                    reduced += 1

                result_pickle = cloudpickle.dumps(accum)
                self._redis.set(sync_key + '_result', result_pickle)
                self._redis.publish(sync_key + '_topic', msgpack.packb({'result_key': sync_key + '_result'}))
                logger.debug('[{}] Notify results of {}'.format(self._worker_id, sync_key))
                sync_result = accum
            elif gatherer:
                pass
                # logger.debug('[{}] Reducing partial results of {}'.format(self._worker_id, key))
                # all_data_pickle = self._redis.lrange(key, 0, index)
                # all_data = [cloudpickle.loads(data_pickle) for data_pickle in all_data_pickle]
                # if operation == CollectiveOPs.SUM:
                #     result = reduce(lambda x, y: x + y, all_data)
                # else:
                #     raise Exception('Unknown operation {}'.format(operation))
                # result_pickle = cloudpickle.dumps(result)
                # self._redis.set(key + '_result', result_pickle)
                # self._redis.publish(key + '_topic', msgpack.packb({'result_key': key + '_result'}))
                # logger.debug('[{}] Notify results of {}'.format(self._worker_id, key))
            else:
                pass
        else:
            data_pickle = cloudpickle.dumps(data)
            self._redis.lpush(sync_key, data_pickle)

            self._redis_pubsub.subscribe(sync_key + '_topic')
            raw_msg = None
            while not raw_msg:
                raw_msg = self._redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=5)
                # print(raw_msg)
            if 'type' not in raw_msg or raw_msg['type'] != 'message':
                raise Exception(raw_msg)
            msg = msgpack.unpackb(raw_msg['data'])
            result_pickle = self._redis.get(msg['result_key'])
            sync_result = cloudpickle.loads(result_pickle)

        return sync_result


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
                func_args = cloudpickle.loads(args_pickle)
                func_args['kwargs']['compute_group'] = task_group_proxy
                logger.debug('[{}] Going to execute task {}'.format(id, task.task_id))
                result = f(data_chunk, *func_args['args'], **func_args['kwargs'])
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
