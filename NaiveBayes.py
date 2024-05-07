import pandas as pd
import numpy as np
import csv
import sys


# This part is to implement the Naive Bayes algorithm, and evaluate the program on the breast cancer
# dataset to be described below. The program should build a Naive Bayes classifier from the training
# dataset and apply it to the test set.

# Training, input the training data and the class labels, and output the prior probabilities and the

class NaiveBayesClassifier:
    
    def __init__(self):
        self.class_prob = {}
        self.conditional_prob = {}

    def fit(self, file):
        with open(file, 'r') as f:
            read = csv.reader(f)
            data = list(read)[1:]
            y_data = [d[-1] for d in data]
            x_data = [d[:-1] for d in data]
            y_counts = self.count(y_data)

        # initialise the class and conditional probabilities
        for label in (y_data):
            self.class_prob[label] = 1
            self.conditional_prob[label] = {}
            for i in range(len(x_data[0])):
                self.conditional_prob[label][i] = {}
                for x in x_data:
                    self.conditional_prob[label][i][x[i]] = 1
        
        # count number of each class and feature
        for x, y in zip(x_data, y_data):
            self.class_prob[y] += 1
            for i , val in enumerate(x):
                self.conditional_prob[y][i][val] += 1

        # calculate the probabilities
        class_total = sum(self.class_prob.values())
        for y, count_y in self.class_prob.items():
            self.class_prob[y] = count_y / class_total
            for i in range(len(x_data[0])):\
                total_i_y = sum(self.conditional_prob[y][i].values()
                for x in self.conditional_prob[y][i].keys():
                    self.conditional_prob[y][i][val] /= total_i_y

                    