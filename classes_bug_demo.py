import numpy as np
from sklearn.svm import SVC

# same dataset
X = np.array([[0.29166667, 0.19366939],
              [0.28611111, 0.34763765],
              [0.26111111, 0.55716146],
              [0.73333333, 0.2016059 ],
              [0.70694444, 0.3952567 ],
              [0.66666667, 0.61271701]])

# two different labelling
y0 = np.array([0, 0, 0, 1, 1, 1])
y1 = np.array([0, 0, 1, 0, 1, 1])

# use a seed for reproducibility
SEED = 0

# train a classifier for y0
np.random.seed(SEED)
clf0 = SVC(gamma='scale', kernel='rbf', probability=True)
clf0.fit(X, y0)

print('####')
print('Classifer 0 summary:')
print('True     labels: {}'.format(y0))
print('Predited labels: {}'.format(clf0.predict(X)))
print('Classification accuracy: {}'.format(clf0.score(X, y0)))
print('--')
print('Classes: {}'.format(clf0.classes_))
print('Predited probabilitic labeling:')
print(clf0.predict_proba(X))


## here the classes ordering matches with the probabilistic prediction

# ####
# Classifer 0 summary:
# True     labels: [0 0 0 1 1 1]
# Predited labels: [0 0 0 1 1 1]
# Classification accuracy: 1.0
# --
# Classes: [0 1]
# Predited probabilitic labeling:
# [[0.98456235 0.01543765]
#  [0.98680646 0.01319354]
#  [0.98189501 0.01810499]
#  [0.0141697  0.9858303 ]
#  [0.01153142 0.98846858]
#  [0.02285266 0.97714734]]


# train a classifier for y1
np.random.seed(SEED)
clf1 = SVC(gamma='scale', kernel='rbf', probability=True)
clf1.fit(X, y1)

print('####')
print('Classifer 1 summary:')
print('True     labels: {}'.format(y1))
print('Predited labels: {}'.format(clf1.predict(X)))
print('Classification accuracy: {}'.format(clf1.score(X, y1)))
print('--')
print('Classes: {}'.format(clf1.classes_))
print('Predited probabilitic labeling:')
print(clf1.predict_proba(X))

## here the classes ordering does not match with the probabilistic prediction

# ####
# Classifer 1 summary:
# True     labels: [0 0 1 0 1 1]
# Predited labels: [0 0 1 0 1 1]
# Classification accuracy: 1.0
# --
# Classes: [0 1]
# Predited probabilitic labeling:
# [[0.0774986  0.9225014 ]
#  [0.21423451 0.78576549]
#  [0.61478601 0.38521399]
#  [0.32195121 0.67804879]
#  [0.69766737 0.30233263]
#  [0.9182796  0.0817204 ]]


## for the same data, with different labels, same seed, the classifier returns probabilisitic prediction in a different order which is not reflected in clf.classes_.

## suggested temporary workaround
def get_ordered_classes(clf, X, y):

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)
    index_pred = np.argmax(y_pred_proba, axis=1)

    ordered_classes = []
    n_class = y_pred_proba.shape[1]
    for class_index in range(n_class):
        # check which data points have been classified as class_index by the probability predictor
        X_indexes_for_class_index = np.where(index_pred == class_index)
        # find what label it corresponds to from the simple predictor
        class_name = y_pred[X_indexes_for_class_index]
        # make sure this is consistent and a unique label is in that list
        # a rare case coudl happen if probab are exaclty equal betwen all classes [0.5 0.5] but so unlikelly that I won't cover it
        class_name = np.unique(class_name)
        assert class_name.shape == (1,), 'Something might be wrong between predict and predict proba in SVC'
        # add class to list
        ordered_classes.append(class_name[0])

    return np.array(ordered_classes)

print('#### classes for y0')
print('classes_ from SVC:                {}'.format(clf0.classes_))
print('classes as used in predict_proba: {}'.format(get_ordered_classes(clf0, X, y0)))

# #### classes for y0
# classes_ from SVC:                [0 1]
# classes as used in predict_proba: [0 1]

print('#### classes for y1')
print('classes_ from SVC:             {}'.format(clf1.classes_))
print('classes used in predict_proba: {}'.format(get_ordered_classes(clf1, X, y1)))

# #### classes for y1
# classes_ from SVC:             [0 1]
# classes used in predict_proba: [1 0]


## Solution using CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV

calibrator0 = CalibratedClassifierCV(clf0, method='sigmoid', cv='prefit')
calibrator0.fit(X, y0)

print('#### classes for y0')
print('classes_ from SVC:                {}'.format(clf0.classes_))
print('classes as used in predict_proba: {}'.format(get_ordered_classes(clf0, X, y0)))
print('classes as used in Calibrator: {}'.format(get_ordered_classes(calibrator0, X, y0)))

# #### classes for y0
# classes_ from SVC:                [0 1]
# classes as used in predict_proba: [0 1]
# classes as used in Calibrator: [0 1]


calibrator1 = CalibratedClassifierCV(clf1, method='sigmoid', cv='prefit')
calibrator1.fit(X, y1)

print('#### classes for y1')
print('classes_ from SVC:             {}'.format(clf1.classes_))
print('classes used in predict_proba: {}'.format(get_ordered_classes(clf1, X, y1)))
print('classes as used in Calibrator: {}'.format(get_ordered_classes(calibrator1, X, y1)))

# #### classes for y1
# classes_ from SVC:             [0 1]
# classes used in predict_proba: [1 0]
# classes as used in Calibrator: [0 1]


print(clf0.predict_proba(X))
print(calibrator0.predict_proba(X))

clf0 = SVC(gamma='scale', kernel='rbf')
calibrator0 = CalibratedClassifierCV(clf0, method='sigmoid', cv=3)
calibrator0.fit(X, y0)

print(calibrator0.predict_proba(X))
