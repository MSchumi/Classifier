# -*- coding:utf-8 -*-
import math
from .helper import RedisDict, FeatureData


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


class BayesClassifier(BaseClassifier):
    def __init__(self, getfeatures, redis):
        super(BayesClassifier, self).__init__(getfeatures, redis)

    def docprob(self, features, cat):
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, features, cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(features, cat)
        return docprob*catprob

    def classify(self, item):
        probs = {}
        max_prob = 0.0
        features = self.getfeatures(item)
        for cat in self.categories():
            probs[cat] = self.prob(features, cat)
            if probs[cat] > max_prob:
                max_prob = probs[cat]
                best = cat
        return best


class FisherClassifier(BaseClassifier):
    def __init__(self, getfeatures, redis):
        super(FisherClassifier, self).__init__(getfeatures, redis)

    def cprob(self, f, cat):
        clf = self.fprob(f, cat)
        if clf == 0:
            return 0
        fresum = sum([self.fprob(f, c) for c in self.categories()])
        p = clf/(fresum)
        return p

    def classify(self, item, default=None):
        best = default
        max_prob = 0.0
        for cat in self.categories():
            p = self.fisherprob(item, cat)
            if p > max_prob:
                best = cat
                max_prob = p
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
