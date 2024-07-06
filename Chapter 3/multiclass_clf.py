from mnist_dataset import *

# multiclass classifiers or multinomial classifiers
# Some Scikit-Learn classifiers (e.g., LogisticRegression, RandomForestClassifier, and GaussianNB)
# are capable of handling multiple classes natively.
# Others are strictly binary classifiers (e.g., SGDClassifier and SVC).

# multiclass problems with binary classifiers -->
# one-versus-the-rest (OvR) strategy, or sometimesone-versus-all (OvA)
# one-versus-one (OvO) strategy : not suitable for many classes : 
# If there are N classes, you need to train N × (N – 1) / 2 classifiers.

from sklearn.svm import SVC 
svm_clf = SVC(random_state =42)
svm_clf .fit (X_train [:2000 ], y_train [:2000 ]) # y_train, not y_train_5

# Since there are 10 classes (i.e., more than 2),
# Scikit-Learn used the OvO strategy and trained 45 binary classifiers

svm_clf .predict ([some_digit ]) # array(['5'], dtype=object)
#That’s correct! This code actually made 45 predictions
# —one per pair of classes—and it selected the class that won the most duels
some_digit_scores = svm_clf .decision_function ([some_digit ])
some_digit_scores .round (2)
# The highest score is 9.3, and it’s indeed the one corresponding to class 5
class_id = some_digit_scores .argmax ()
class_id # 5
#When a classifier is trained, it stores the list of target classes 
# in its classes_ attribute, ordered by value.
svm_clf .classes_ # array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
svm_clf .classes_ [class_id ] # here class id and class label are same 

# using OvR strategy
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier (SVC (random_state =42))
ovr_clf .fit (X_train [:2000 ], y_train [:2000 ])

# check the number of trained classifiers
ovr_clf .predict ([some_digit ])# array(['5'], dtype='<U1')
len(ovr_clf .estimators_ )# 10

# SGDClassifier on multiclass data set 
sgd_clf = SGDClassifier (random_state =42) 
sgd_clf .fit(X_train , y_train ) 
sgd_clf .predict ([some_digit ]) # array(['3'], dtype='<U1') ( incorrect prediction, would be 5)

# look at the scores that the SGD classifier assigned to each class
sgd_clf .decision_function ([some_digit ]).round ()
# cross v 
cross_val_score (sgd_clf , X_train , y_train , cv=3, scoring ="accuracy" )# array([0.87365, 0.85835, 0.8689 ])
# scaling ( see chapter 2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler () 
X_train_scaled = scaler .fit_transform (X_train .astype ("float64" ))
cross_val_score (sgd_clf , X_train_scaled , y_train , cv=3, scoring ="accuracy" ) #array([0.8983, 0.891 , 0.9018])

# error alaysis
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf , X_train_scaled , y_train , cv=3)
ConfusionMatrixDisplay .from_predictions (y_train , y_train_pred )
plt.show ()
# normalize by row , showing percent 
ConfusionMatrixDisplay .from_predictions (y_train , y_train_pred , normalize ="true" , values_format =".0%" )
plt.show ()
# error matrix
sample_weight = (y_train_pred != y_train )
ConfusionMatrixDisplay .from_predictions (y_train , y_train_pred ,
                    sample_weight =sample_weight , normalize ="true" , values_format =".0%" )
plt.show ()