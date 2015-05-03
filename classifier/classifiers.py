# -*- coding:utf-8 -*-
import math

PREFIX = "bayes:"


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


class BaseClassifier(object):
    def __init__(self, getfeatures, redis):
        self.fc = FeatureData(redis, "feature:")
        self.cc = RedisDict(redis, "category")
        self.getfeatures = getfeatures

    def incf(self, feature, category):
        self.fc.setdefault(feature, category, 0)
        self.fc.incrby(feature, category, 1)

    def incc(self, category):
        self.cc.setdefault(category, 0)
        self.cc.incrby(category, 1)

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        return sum([int(x) for x in self.cc.values()])

    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        return self.fcount(f, cat)/self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        basicprob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = (weight*ap+(totals*basicprob))/(weight+totals)
        return bp

    def sampletrain(self):
        self.train("Nobody owns the water", 'good')
        self.train('the quick rabbit jumps fences', 'good')
        self.train('buy pharmaceuticals now', 'bad')
        self.train('make quick money at the online casino', 'bad')
        self.train('the quick brown fox jumps', 'good')


class BayesClassifier(BaseClassifier):
    def __init__(self, getfeatures, redis):
        super(BayesClassifier, self).__init__(getfeatures, redis)
        self.thresholds = RedisDict(redis, "thresholds")

    def docprob(self, item, cat):
        features = self.getfeatures(item)
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob*catprob

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        return float(self.thresholds[cat])

    def classify(self, item, default=None):
        probs = {}
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                return default
        return best


class FisherClassifier(BaseClassifier):
    def __init__(self, getfeatures, redis):
        super(FisherClassifier, self).__init__(getfeatures, redis)
        self.minimums = RedisDict(redis, "minimums")

    def setminmum(self, cat, min):
        self.minimums[cat] = min

    def getminmum(self, cat):
        if cat not in self.minimums:
            return 0
        return self.minimums[cat]

    def cprob(self, f, cat):
        clf = self.fprob(f, cat)
        if clf == 0:
            return 0
        fresum = sum([self.fprob(f, c) for c in self.categories()])
        p = clf/(fresum)
        return p

    def classify(self, item, default=None):
        best = default
        max = 0.0
        for c in self.categories():
            p = self.fisherprob(item, c)
            if p > self.getminmum(c) and p > max:
                best = c
                max = p
        return best

    def fisherprob(self, item, cat):
        p = 1
        features = list(self.getfeatures(item))
        for f in features:
            p *= self.weightedprob(f, cat, self.cprob)
        fscore = -2 * math.log(p)
        return self.invchi2(fscore, len(features)*2)

    def invchi2(self, chi, df):
        m = chi/2.0
        sum = term = math.exp(-m)
        for i in range(1, df//2):
            term *= m/i
            sum += term
        return min(sum, 1.0)

if __name__ == "__main__":
    pass
    # import redis
    # getwords = get_ch_words
    # redis = redis.StrictRedis(db=1)
    # c1=BaseFilter(getwords, redis)
    # c1.sampletrain()
    # print c1.fcount('quick','bad')
    # print c1.fcount('quick','good')
    # print c1.fprob('quick','good')
    # print c1.weightedprob('money','good',c1.fprob)
    # print "###########"
    # c2 = BayesFilter(getwords,redis)
    # #import pdb; pdb.set_trace()
    # #c2.sampletrain()
    # print c2.classify('quick rabbit', default='unknow')
    # print c2.classify('quick money', default='unknow')
    # print c2.setthreshold('bad', 3.0)
    # print c2.classify('quick money', default='unknow')
    # #for i in range(10):
    #     #c2.sampletrain()
    # print c2.classify('quick money', default='unknow')
    # #print c2.prob('quick rabbit', 'good')
    # #print c2.prob('quick rabbit', 'bad')
    # print "#############"
    # c3 = FisherFilter(getwords, redis)
    # #c3.sampletrain()
    # print c3.cprob('quick', 'good')
    # print c3.fisherprob('quick rabbit','good')
    # print c3.classify('quick money')
    # print c3.classify('quick rabbit')
    # print c3.setminmum('bad', 0.8)
    # print c3.classify('quick money')
    # print c3.setminmum('good', 0.4)
    # print c3.classify('quick money')
    # print "#################"

