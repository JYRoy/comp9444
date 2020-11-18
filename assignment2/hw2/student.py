#!/usr/bin/env python3
# encoding: utf-8
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import string
import re
from config import device


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    processed = sample.split()
    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising. "numericalising" is the process of assigning each word (or token) in the text with a number or id
    """
    # remove illegal characters
    sample = [re.sub(r'[^\x00-\x7f]', r'', word) for word in sample]
    # remove punctuations
    sample = [word.strip(string.punctuation) for word in sample]
    # remove numbers
    sample = [re.sub(r'[0-9]', r'', word) for word in sample]
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising. "vectorising" is when we transform a text into a vector 
    """
    return batch


# Useless words.
stopWords = {'i', 'oh', "i'm", "i've", "i'd", "i'll", 'me', 'my', 'myself', 'we', "we've", "we'd", "we'll", 'us',
             'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
             'yourselves', 'he', "he'll", "he'd", 'him', 'his', 'himself', 'she', "she'll", "she'd", "she's",
             'her', 'hers', 'herself', 'it', "it'll", "it's", 'its', 'itself', 'they', "they're", "they'll",
             'them', 'their', 'theirs', 'themselves', 'what', "what's", 'which', 'who', 'whom', 'this', 'that',
             "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
             'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
             'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
             'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
             'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
             'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
             "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
             'couldn', 'could', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
             'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
             'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", "would",
             'wouldn', "wouldn't", 'yep', 'co', "food", "restaurant", "place", "day", "fees", "bank"}

wordVectors = GloVe(name='6B', dim=200)  # name of the file that contains the vectors把原本的单词转化成vector  调用GloVe 50维的


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.

    针对Calculate performance部分的问题,数据格式（LongTensor），数据维度
    correctRating = rating == ratingOutputs.flatten()
    correctCategory = businessCategory == categoryOutputs.flatten()
    """
    ratingOutput = ratingOutput.long()
    categoryOutput = categoryOutput.long()
    return ratingOutput.argmax(dim=1), categoryOutput.argmax(dim=1)


################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.Relu = tnn.ReLU()
        self.dropout = tnn.Dropout(0.5)
        self.lstm1 = tnn.LSTM(
            input_size=200,
            hidden_size=400,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True)
        self.linear_rating_1 = tnn.Linear(in_features=400, out_features=200)
        self.linear_rating_3 = tnn.Linear(in_features=200, out_features=2)

        self.lstm2 = tnn.LSTM(
            input_size=200,
            hidden_size=400,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True)
        self.linear_category_1 = tnn.Linear(in_features=400, out_features=200)
        self.linear_category_3 = tnn.Linear(in_features=200, out_features=5)

    def forward(self, input, length):
        output_rating, (hidden_rating, cell_rating) = self.lstm1(input)
        hidden_rating = hidden_rating[-1, :, :]
        ratingOutput = self.Relu(self.dropout(self.linear_rating_1(hidden_rating)))
        ratingOutput = self.linear_rating_3(ratingOutput)

        output_category, (hidden_category, cell_category) = self.lstm2(input)
        hidden_category = hidden_category[-1, :, :]
        categoryOutput = self.Relu(self.dropout(self.linear_category_1(hidden_category)))
        categoryOutput = self.linear_category_3(categoryOutput)
        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.entroy = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        lossRating = self.entroy(ratingOutput, ratingTarget)
        lossCategory = self.entroy(categoryOutput, categoryTarget)
        return lossRating + lossCategory


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 128
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.003)
