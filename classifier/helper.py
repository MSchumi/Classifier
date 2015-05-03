# -*- coding:utf-8 -*-


class RedisDict(object):
    def __init__(self, redis, hkey):
        self.redis = redis
        self.hkey = hkey

    def __getitem__(self, key):
        if not self.redis.hexists(self.hkey, key):
            return None
        else:
            return self.redis.hget(self.hkey, key)

    def __setitem__(self, key, value):
        self.redis.hset(self.hkey, key, value)

    def setdefault(self, key, value):
        if not self.redis.hexists(self.hkey, key):
            self.redis.hset(self.hkey, key, value)

    def keys(self):
        if not self.redis.exists(self.hkey):
            return []
        return self.redis.hkeys(self.hkey)

    def values(self):
        if not self.redis.exists(self.hkey):
            return []
        return self.redis.hvals(self.hkey)

    def __contains__(self, key):
        return self.redis.hexists(self.hkey, key)

    def incrby(self, key, value=1):
        if self.redis.hexists(self.hkey, key):
            self.redis.hincrby(self.hkey, key, value)


class FeatureData(object):
    def __init__(self, redis, prefix):
        self.redis = redis
        self.prefix = prefix

    def exists_feature(self, feature):
        return self.redis.exists(self.prefix+feature)

    def setdefault(self, feature, category, value):
        if not self.redis.hexists(self.prefix+feature, category):
            self.redis.hset(self.prefix+feature, category, value)

    def incrby(self, feature, category, value=1):
        if self.redis.hexists(self.prefix+feature, category):
            self.redis.hincrby(self.prefix+feature, category, value)

    def __getitem__(self, feature):
        if self.exists_feature(feature):
            return RedisDict(self.redis, self.prefix+feature)
        return None

    def __contains__(self, feature):
        return self.exists_feature(feature)
