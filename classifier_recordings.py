from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import csv
import numpy as np
import itertools
import pickle

# labels of classes
vocalization_labels = ['babbling', 'crying', 'jargon', 'nonarticulated', 'speech']

# for drawing
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('figure.pdf', bboxinches='tight')
    plt.show()

# open the results of feature selection
with open("classifier_selected_features.sav", 'rb') as f:
    classifier, feature_mask = pickle.load(f)

classes = []
features = []
feature_names = ['zcr', 'hzcrr', 'kurtosis', 'skewness', 'spectral mean', 'spectral_variance', 'spectral deviation',
                 'spectral centroid', 'spectral kurtosis', 'spectral skewness', 'loudness', 'rms', 'lpc1', 'lpc2',
                 'lpc3', 'lpc4', 'lpc5', 'lpc6', 'lpc7', 'lpc8', 'lpc9', 'lpc10', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
                 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'bark1', 'bark2', 'bark3', 'bark4', 'bark5',
                 'bark6', 'bark7', 'bark8', 'bark9', 'bark10', 'bark11', 'bark12', 'bark13', 'bark14', 'bark15',
                 'bark16', 'bark17', 'bark18', 'bark19', 'bark20', 'bark21', 'bark22', 'bark23', 'bark24']

# open results of feature calculation on a dataset
with open("features_nao_cleaned.csv", 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        classes.append(row[0])
        feature_row = []
        for i in range(len(feature_mask)):
            if feature_mask[i]:
                feature_row.append(row[i+1])
        features.append(feature_row)

# generate predictions with classifier
predictions = classifier.predict(features)

# generate confusion matrix
conf_matrix_test = 1.0*confusion_matrix(classes, predictions)

# do calculation and output
TP = [conf_matrix_test[x, x] for x in range(5)]
FP = [(np.sum(conf_matrix_test, axis=0)[i]) - conf_matrix_test[i, i] for i in range(5)]
FN = [(np.sum(conf_matrix_test, axis=1)[i]) - conf_matrix_test[i, i] for i in range(5)]
TN = [np.sum(conf_matrix_test) - (np.sum(conf_matrix_test, axis=1)[i]) - (np.sum(conf_matrix_test, axis=0)[i]) + conf_matrix_test[i,i] for i in range(5)]

print(TP)
print(FP)
print(FN)
print(TN)

accuracy = [(TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]) for i in range(5)]
precision = [TP[i]/(TP[i]+FP[i]) for i in range(5)]
recall = [TP[i]/(TP[i]+FN[i]) for i in range(5) ]

print(precision)
print(recall)
print(accuracy)
DOR = [(TP[i]/FP[i])/(FN[i]/TN[i]) for i in range(5)]
print(DOR)

# do micro averaging
precision_micro_avg = np.sum(np.asarray(TP))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FP)))
recall_micro_avg = np.sum(np.asarray(TP))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FN)))
accuracy_micro_avg = (np.sum(np.asarray(TP))+np.sum(np.asarray(TN)))/(np.sum(np.asarray(TP))+np.sum(np.asarray(FN))+np.sum(np.asarray(TN))+np.sum(np.asarray(FP)))
dor_micro_avg = (np.sum(np.asarray(TP))/np.sum(np.asarray(FP)))/(np.sum(np.asarray(FN))/np.sum(np.asarray(TN)))

# print results of micro averaging
print(precision_micro_avg, recall_micro_avg, accuracy_micro_avg, dor_micro_avg)

# plot results on the test part in form of a confusion matrix
plot_confusion_matrix(conf_matrix_test, vocalization_labels, normalize=True)

