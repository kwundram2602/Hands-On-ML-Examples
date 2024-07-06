#Occasionally you will need more control over the cross-validation process#
# than what Scikit-Learn provides off the shelf.
# In these cases, you can implement cross-validation yourself.
# The following code does roughly the same thing as Scikit-Learn’scross_val_score() function, 
# and it prints the same result:

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold (n_splits =3)
# look in mnist_dataset.py
# add shuffle=True if the dataset is # not already shuffled
for train_index , test_index in skfolds .split (X_train , y_train_5 ):
    clone_clf = clone (sgd_clf )
    X_train_folds = X_train [train_index ]
    y_train_folds = y_train_5 [train_index ]
    X_test_fold = X_train [test_index ]
    y_test_fold = y_train_5 [test_index ]
    clone_clf .fit(X_train_folds , y_train_folds )
    y_pred = clone_clf .predict (X_test_fold )
    n_correct = sum(y_pred == y_test_fold )
    print (n_correct / len (y_pred ))
    
#The StratifiedKFold class performs stratified sampling (as explained in Chapter 2)
# to produce folds that contain a representative ratio of each class.
# At each iteration the code creates a clone of the classifier,
# trains that clone on the training folds, and makes predictions on the test fold.
# Then it counts the number of correct predictions and outputs the ratio of correct predictions.