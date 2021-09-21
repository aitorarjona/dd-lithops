import lithops


def extract_redis_config(**overwrites):
    redis_conf = lithops.config.load_config()['redis']
    redis_conf.update(overwrites)
    return redis_conf


def clean_keys(shards, storage=None):
    keys = [shard.key for shard in shards]
    store = storage or lithops.storage.Storage()
    store.delete_objects(bucket=store.bucket, key_list=[shard.key for shard in shards])


def partition_list_generator(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

