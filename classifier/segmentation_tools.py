# -*- coding:utf-8 -*-


def jieba_seg(text):
    from jieba.analyse import extract_tags
    topk = 30
    return extract_tags(text, topk)


def getwords(doc):
    import re
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and
             len(s) < 20]
    return dict([(w, 1) for w in words])


def get_ch_words(doc):
    import jieba
    seg_list = jieba.cut(doc)
    return filter(lambda x: x.strip(), seg_list)
