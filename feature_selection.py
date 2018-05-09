from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import csv
import pickle
classes = []
features = []
feature_names = ['zcr', 'hzcrr', 'kurtosis', 'skewness', 'spectral mean', 'spectral_variance', 'spectral deviation',
                 'spectral centroid', 'spectral kurtosis', 'spectral skewness', 'loudness', 'rms', 'lpc1', 'lpc2',
                 'lpc3', 'lpc4', 'lpc5', 'lpc6', 'lpc7', 'lpc8', 'lpc9', 'lpc10', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
                 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'bark1', 'bark2', 'bark3', 'bark4', 'bark5',
                 'bark6', 'bark7', 'bark8', 'bark9', 'bark10', 'bark11', 'bark12', 'bark13', 'bark14', 'bark15',
                 'bark16', 'bark17', 'bark18', 'bark19', 'bark20', 'bark21', 'bark22', 'bark23', 'bark24']

# open calculated features dataset
with open("features_8192_512.csv", 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        classes.append(row[0])
        features.append(row[1:-1])

# create classifier
clf = RandomForestClassifier(n_estimators=len(features[0]), max_depth=20, min_samples_split=5, random_state=None)

# create feature selector (recursive feature elimination with cross validation)
rfecv = RFECV(estimator=clf, step=1)

# select features and train classifier
rfecv.fit(features, classes)

# print results
print(rfecv.n_features_)
print(rfecv.support_)
print(rfecv.ranking_)
print(rfecv.estimator_)

# extract classifier from feature selection
classifier = (rfecv.estimator_, rfecv.support_)

# save classifier and feature mask
with open("classifier_selected_features.sav", 'wb') as f:
    pickle.dump(classifier, f)

