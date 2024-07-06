from mnist_dataset import *

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.svm import SVC 
y_train_large = (y_train >= '7') 
y_train_odd = (y_train .astype ('int8' ) % 2 == 1)
# creating multilabel array containing two target labels
y_multilabel = np.c_[y_train_large , y_train_odd ]
knn_clf = KNeighborsClassifier ()
knn_clf .fit (X_train , y_multilabel )

knn_clf .predict ([some_digit ]) #array([[False, True]]) digit 5 is indeed not large ( False) and odd (True)
# compute average f1 scroe for all labels
y_train_knn_pred = cross_val_predict (knn_clf , X_train , y_multilabel , cv=3)
f1_score (y_multilabel , y_train_knn_pred , average ="macro" ) # average="weighted" for respecting number of instances with that target label

# chain classifier using cross validation
# usiing  the first 2,000 images in the training set
from sklearn.multioutput import ClassifierChain
chain_clf = ClassifierChain (SVC(), cv=3, random_state =42)
chain_clf .fit(X_train [:2000 ], y_multilabel [:2000 ])