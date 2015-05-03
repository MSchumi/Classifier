#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Manager tool for filters

    Usage:
        manager.py train [--dir=<dir>] [--category=<category>] [--count=<count>]
        manager.py test [--dir=<dir>] [--classifier=<classifier>] [--start=<start>] [--count=<count>]
        manager.py (-h | --help)
    Options:
        -h,--help                       Show this info
        -d,--dir <dir>                  Doc directory
        -c,--category <category>        The doc category
        --count <count>
        -f,--classifier <classifier>    Default bayes
        -s,--start <start>              Index
"""

from docopt import docopt
from classifier import classifiers, segmentation_tools
import redis
import os

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
        start = args["--start"]
        bayes_test(dirname, f_type, start=start, count=count)

def train(dirname, category, count=100):
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


def bayes_test(dirname, f_type, start=0, count=100):
    redis = redis.StrictRedis(db=1)
    get_word = segmentation_tools.jieba_seg
    if f_type == "fisher":
        print "fisher"
        classifier = classifiers.FisherClassifier(get_word, redis)
    else:
        classifier = classifiers.BayesClassifier(get_word, redis)
    files = os.listdir(dirname)
    classes = {}
    if len(files) > start + 100:
        files = files[start: start + count]
    for f in files:
        if f.rfind(".txt") == -1:
            continue
        temp = open(os.path.join(dirname, f), "r")
        text = temp.read()
        cat = classifier.classify(text)
        classes.setdefault(cat, 0)
        classes[cat]+=1
    print classes


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
