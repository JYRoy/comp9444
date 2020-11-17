#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

import numpy as np

if __name__ == '__main__':
    X = np.array([[[1, 1.1], [2, 2.2], [3, 3.3]], [[4, 4.4], [5, 5.5], [6, 6.6]]])
    print(X[-1, -1, -1])
    import nltk

    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words("english")
    print(len(stopwords))
    print(stopwords)
