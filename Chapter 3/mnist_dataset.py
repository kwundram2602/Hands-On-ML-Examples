from sklearn.datasets import fetch_openml
#load_* functions to load small toy datasets
#fetch_* functions such asfetch_openml() to download real-life datasets,
#and make_* functions to generate fake datasets
#. Generated datasets are usually returned as an (X, y) tuple containing the input data and the target
mnist = fetch_openml ('mnist_784' , as_frame =False )
X, y = mnist .data , mnist .target

import matplotlib.pyplot as plt

def plot_digit (image_data ):
    image = image_data .reshape (28, 28)
    plt.imshow (image , cmap ="binary" ) 
    plt.axis ("off" ) 
some_digit = X[0]
X_train , X_test , y_train , y_test = X[:60000 ], X[60000 :], y[:60000 ], y[60000 :]

y_train_5 = (y_train == '5' )
y_test_5 = (y_test == '5')


#SGDClassifier class.
# This classifier is capable of handling very large datasets efficiently.
# This is in part because SGD deals with training instances independently,
# one at a time, which also makes SGD well suited for online learning

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier (random_state =42)
sgd_clf.fit (X_train , y_train_5 )
sgd_clf.predict ([some_digit ])

# k fold cross validation
# evaluate our SGDClassifier model, using k-fold cross-validation with three folds
#means splitting the training set intok folds (in this case, three),
# then training the modelk times, holding out a different fold each time for evaluation
from sklearn.model_selection import cross_val_score
cross_val_score (sgd_clf , X_train , y_train_5 , cv=3, scoring ="accuracy" )

# dummmy classifeier  and cv
#dummy classifier that just classifies every single image
# in the most frequent class, which in this case is the negative class
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier ()
dummy_clf .fit(X_train , y_train_5 )
cross_val_score (dummy_clf , X_train , y_train_5 , cv=3, scoring ="accuracy" )
#A much better way to evaluate the performance of a classifier is to look at the confusion matrix (CM) !!!


if __name__ == "__main__":
    plot_digit (some_digit )
    plt.show ()
    print(y[0])
    # dummy classifier
    print (any(dummy_clf .predict (X_train )))

