import logging
import io

import lithops
import numpy as np

from lithopsext.datasetbase import DatasetBase

logger = logging.getLogger('lithops')


class PartitionedArray(DatasetBase):
    def __init__(self):
        raise Exception('Dont instantiate this class directly, use factory methods instead')

    @classmethod
    def from_numpy(cls, arr, partitions):
        instance = cls.__new__(cls)
        super().__init__(instance)
        instance._partition_numpy_shards(arr, partitions)
        return instance

    def _partition_numpy_shards(self, arr, partitions):
        for i, chunk in enumerate(np.split(arr, partitions)):
            partition = _NumpyArrayPartition(self._lithops_storage, self._group_id, i, chunk)
            self._partitions.append(partition)


class _NumpyArrayPartition:
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
