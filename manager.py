#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Manager tool for filters

    Usage:
        manager.py train [--dir=<dir>] [--category=<category>] [--count=<count>]
        manager.py test [--dir=<dir>] [--filter=<filter>] [--count=<count>]
        manager.py (-h | --help)
    Options:
        -h,--help                       Show this info
        -d,--dir <dir>                  Doc directory
        -c,--category <category>        The doc category
        --count <count>
        -f,--filter <filter>            Default bayes
"""

from docopt import docopt
from classifier import classifiers, segmentation_tools


def main(args):
    if args["train"]:
        dirname = args["--dir"]
        category = args["--category"]
        count = args["--count"]
        train(dirname, category, count=count)
    if args["test"]:
        dirname = args["--dir"]
        f_type = args["--filter"]
        count = args["--count"]
        bayes_test(dirname, f_type, count=count)

def train(dirname, category, count=100):
    import redis
    import os
    if not count:
        count = 100
    count = int(count)
    redis = redis.StrictRedis(db=1)
    get_word = segmentation_tools.jieba_seg
    basefilter = classifiers.BaseClassifier(get_word, redis)
    files = os.listdir(dirname)
    if len(files) > count:
        files = files[:count]
    for f in files:
        if f.rfind(".txt") == -1:
            continue
        temp = open(os.path.join(dirname, f), "r")
        text = temp.read()
        temp.close()
        basefilter.train(text, category)


def bayes_test(dirname, f_type, count=100):
    import redis
    import os
    import time
    now = time.time()
    if not count:
        count = 100
    count = int(count)
    redis = redis.StrictRedis(db=1)
    get_word = segmentation_tools.jieba_seg
    if f_type == "fisher":
        print "fisher"
        classifier = classifiers.FisherClassifier(get_word, redis)
    else:
        classifier = classifiers.BayesClassifier(get_word, redis)
    files = os.listdir(dirname)
    b_classes = {}
    if len(files) > count+100:
        files = files[count:count+100]
    for f in files:
        if f.rfind(".txt") == -1:
            continue
        temp = open(os.path.join(dirname, f), "r")
        text = temp.read()
        cat = classifier.classify(text)
        b_classes.setdefault(cat, 0)
        b_classes[cat]+=1
    print b_classes
    print time.time()-now
        


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
