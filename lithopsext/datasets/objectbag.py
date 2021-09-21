import fnmatch
import random

from lithopsext.datasetbase import DatasetBase


class ObjectBag(DatasetBase):
    def __init__(self):
        raise Exception('Dont instantiate this class directly, use factory methods instead')

    @classmethod
    def s3_glob(cls, pattern):
        self = cls.__new__(cls)
        super().__init__()
        self._setup()
        self._s3_glob(pattern)
        return self

    def _s3_glob(self, pattern, batch_size=1000, lazy_loading=True, shuffle=False, sample=None):
        assert pattern.startswith('s3://'), 'Path must be s3://bucket/path'
        bucket, path = pattern.replace('s3://', '').split('/', 1)
        keys = self._lithops_storage.list_keys(bucket=bucket)
        filter_keys = fnmatch.filter(keys, path)
        assert filter_keys, 'No keys match pattern {}!'.format(pattern)

        if sample:
            assert sample > 0, 'Sample value must be greater than 0'
            assert sample <= len(filter_keys), 'Sample value must be less than' \
                                               ' total population (of size {})'.format(len(filter_keys))
            filter_keys = random.sample(filter_keys, sample)
        if shuffle:
            random.shuffle(filter_keys)

        batches = utils.partition_list_generator(filter_keys, batch_size)

        self._shards = [
            _ObjectBagShard(self._lithops_storage, self._group_id, i, )
        ]


class _ObjectBagShard:
    def __init__(self, storage, group_id, part, chunk, lazy_loading):
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
