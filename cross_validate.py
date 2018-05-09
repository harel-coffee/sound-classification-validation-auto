#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, precision_score

vocalization_labels = ['babbling', 'crying', 'jargon', 'nonarticulated', 'speech']

def plot_confusion_matrix(cm, classes, data_type, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix. Normalization can be applied by setting 'normalize=True'."""


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        print('Normalized confusion matrix for {} dataset'.format(data_type))
    else:
        print('Confusion matrix for {} dataset, without normalization'.format(data_type))


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def crossValidation(directory, csvfile, type='Train'):
    """The function calculates running average for each confusion matrix element by going through all matrices. The program can be called with 1 or 3 arguments from command line. The first argument is 'test' or 'train' which indicates which matrices to calculate moving average for and print them. The second and third arguments are the row and column numbers of matrix which you want to plot. Function then plots the value of running average after every iteration to see the convergence of cross-validation. Example: cross_validation.py train 1 1"""
    cur_avg = np.zeros((5, 5))
    n=0.0
    if len(sys.argv) == 4:
        plt.figure()
        plt.ion()

    for i in range(len(os.listdir(directory))):
        print('Compeleted %s %%' % (100.*(i + 1.0)/len(os.listdir(directory))))
        matrix_list = np.genfromtxt(csvfile+str(i+1)+'.csv', delimiter=',')
        cur_avg = cur_avg + (matrix_list - cur_avg)/(n+1.0)
        n += 1
        if len(sys.argv) == 4:
            plt.scatter(i, cur_avg[sys.argv[2]][sys.argv[3]])

    np.set_printoptions(precision=2)
    plt.figure()
    precision = [cur_avg[i, i]/(np.sum(cur_avg, axis=1)[i]) for i in range(5)]
    recall = [cur_avg[i, i] / (np.sum(cur_avg, axis=0)[i]) for i in range(5)]
    # print(precision)
    # print(recall)
    TP = [cur_avg[x, x] for x in range(5)]
    FP = [(np.sum(cur_avg, axis=0)[i]) - cur_avg[i, i] for i in range(5)]
    FN = [(np.sum(cur_avg, axis=1)[i]) - cur_avg[i, i] for i in range(5)]
    TN = [np.sum(cur_avg) - (np.sum(cur_avg, axis=1)[i]) - (np.sum(cur_avg, axis=0)[i]) + cur_avg[i,i] for i in range(5)]

    print(TP)
    # print(FP)
    # print(FN)
    # print(TN)

    accuracy = [(TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]) for i in range(5)]
    precision = [TP[i]/(TP[i]+FP[i]) for i in range(5)]
    recall = [TP[i]/(TP[i]+FN[i]) for i in range(5) ]

    print(precision)
    print(recall)
    print(accuracy)
    DOR = [(TP[i]/FP[i])/(FN[i]/TN[i]) for i in range(5)]
    print(DOR)

    #micro averaging

    precision_micro_avg = np.sum(np.asarray(TP))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FP)))
    recall_micro_avg = np.sum(np.asarray(TP))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FN)))
    accuracy_micro_avg = (np.sum(np.asarray(TP))+np.sum(np.asarray(TN)))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FN))+np.sum(np.asarray(TN))+np.sum(np.asarray(FP)))
    dor_micro_avg = (np.sum(np.asarray(TP))/np.sum(np.asarray(FP)))/(np.sum(np.asarray(FN))/np.sum(np.asarray(TN)))
    print(precision_micro_avg, recall_micro_avg, accuracy_micro_avg, dor_micro_avg)
    plot_confusion_matrix(cur_avg, classes=vocalization_labels, data_type=type, title='')
    plt.savefig('confusion-test.pdf', bboxinches='tight')
    plt.show()


    if len(sys.argv) == 4:
        while True:
            plt.pause(0.05)


'''Main function'''
if __name__ == '__main__':
    csvfile = './ReducedFeatures/Test/results_test'
    directory = './ReducedFeatures/Test'
    crossValidation(directory, csvfile, type='Test')
