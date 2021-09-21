import redis
import itertools
import hashlib
import cloudpickle
import logging
import threading
import msgpack

import lithops

from lithopsext import utils
from .core import _task_worker

logger = logging.getLogger('lithops')

DEBUG = True


class DatasetBase:
    def __init__(self):
        self._partitions = []
        self._lithops_storage = lithops.storage.Storage()
        self._lithops_executor = lithops.FunctionExecutor()
        self._group_id = '-'.join(['lithops', self._lithops_executor.executor_id, 'group'])
        self._red = redis.Redis(**utils.extract_redis_config())
        self._pool_active = False
        self._task_counter = itertools.count(0)
        self._worker_futures = []

        assert self._red.ping()

    @property
    def _num_partitions(self):
        return len(self._partitions)

    def parallel_apply(self, f, *args, **kwargs):
        task_id = self._group_id + '-' + str(next(self._task_counter)).zfill(3)

        func_pickle = cloudpickle.dumps(f)
        func_pickle_hash = hashlib.md5(func_pickle).hexdigest()
        func_key = '-'.join(['func', f.__name__, func_pickle_hash])

        if not self._red.hexists(self._group_id, func_key):
            self._red.hset(self._group_id, func_key, func_pickle)

        args_pickle = cloudpickle.dumps({'args': args, 'kwargs': kwargs})
        args_key = 'args-{}'.format(task_id)
        self._red.hset(self._group_id, args_key, args_pickle)

        task = {'action': 'task',
                'task_id': task_id,
                'group_size': self._num_partitions,
                'task_join_bl': '{}-bl'.format(task_id),
                'task_join_counter': '{}-cnt'.format(task_id),
                'func_key': func_key,
                'args_key': args_key}

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
                self._worker_futures = []
                for i, shard in enumerate(self._partitions):
                    thread = threading.Thread(target=_task_worker, args=(i, shard, self._group_id))
                    thread.start()
                    self._worker_futures.append(thread)
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
                results = [cloudpickle.loads(res) for res in results_pickle]
                # print(results)
                wait_loop = False
        print(results)
        return results[0]


class PartitionBase:

    @property
    def key(self):
        return None

    def get(self):
        pass
