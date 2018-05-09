from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import csv
import numpy as np
import itertools
import pickle

# load the the results of feature selection
with open("classifier_selected_features.sav", 'rb') as f:
    classifier, feature_mask = pickle.load(f)
# check feature mask
print(feature_mask)
# check classifier parameters
print(classifier)

classes = []
features = []
feature_names = ['zcr', 'hzcrr', 'kurtosis', 'skewness', 'spectral mean', 'spectral_variance', 'spectral deviation',
                 'spectral centroid', 'spectral kurtosis', 'spectral skewness', 'loudness', 'rms', 'lpc1', 'lpc2',
                 'lpc3', 'lpc4', 'lpc5', 'lpc6', 'lpc7', 'lpc8', 'lpc9', 'lpc10', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
                 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'bark1', 'bark2', 'bark3', 'bark4', 'bark5',
                 'bark6', 'bark7', 'bark8', 'bark9', 'bark10', 'bark11', 'bark12', 'bark13', 'bark14', 'bark15',
                 'bark16', 'bark17', 'bark18', 'bark19', 'bark20', 'bark21', 'bark22', 'bark23', 'bark24']

# load data from feature calculation
with open("features_8192_512.csv", 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        classes.append(row[0])
        feature_row = []
        for i in range(len(feature_mask)):
            if feature_mask[i]:
                feature_row.append(row[i+1])
        features.append(feature_row)

# do N iterations of train/test
for i in range(100):
    clf = RandomForestClassifier(n_estimators=len(features[0]), max_depth=20, min_samples_split=5, random_state=None)
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.25, random_state=None)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    conf_matrix_test = confusion_matrix(y_test, predictions)
    conf_matrix_train = confusion_matrix(y_train, clf.predict(X_train))

    # save data
    with open("./ReducedFeatures/Test/results_test%s.csv" % (i+1), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(conf_matrix_test)
    with open("./ReducedFeatures/Train/results_train%s.csv" % (i + 1), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(conf_matrix_train)
    print('Completed %s %%' % (i+1))

