#The general idea of a confusion matrix is to count the number of times instances
# of class A are classified as class B,
# for all A/B pairs. For example:
# to know the number of times the classifier confused images of 8s with 0s,
# you would look at row #8, column #0 ofthe confusion matrix.
#----------------------------------------------------
#To compute the confusion matrix, you first need to have a set of predictions
# so that they can be compared to the actual targets.
# You could make predictions on the test set,
# but it’s best to keep that untouched for now

from mnist_dataset import *
from sklearn.model_selection import cross_val_predict

#performs k-fold cross-validation,
# but instead of returning the evaluation scores,
# it returns the predictions made on each test fold
y_train_pred = cross_val_predict (sgd_clf , X_train , y_train_5 , cv=3)

from sklearn.metrics import confusion_matrix 
# pass target classes (y_train_5)
# and the predicted classes (y_train_pred)
cm = confusion_matrix (y_train_5 , y_train_pred )
# --> array([[53892, 687],
#           [ 1891, 3530]]) 
# 687 false positive = type 1 error
# 1,891 false negative  = type 2 error
#Each row in a confusion matrix represents anactual class
# , while each column represents a predicted class.

# TP = true positives
# FP = false positives 
# FN = False negatives 
#precision = TP /(TP + FP )
# recall = TP / ( TP + FN )
# recall or sensitivity or the true positive rate (TPR)

from sklearn.metrics import precision_score , recall_score
precision_score (y_train_5 , y_train_pred ) 
# == 3530 / (687 + 3530) 0.8370879772350012 
recall_score (y_train_5 , y_train_pred )
# == 3530 / (1891 + 3530) 0.6511713705958311

#F1 score is the harmonic mean of prescision and recall
#Whereas the regular mean treats all values equally,
# the harmonic mean gives much more weight to low values.
# As a result, the classifier will only get a high F1 score if both recall and precision are high
# F1 = (precision * recall) / (precision + recall)

from sklearn.metrics import f1_score
f1_score (y_train_5 , y_train_pred )

#Scikit-Learn does not let you set the threshold directly
#, but it does give you access to the decision scores that it uses to make predictions

y_scores = sgd_clf .decision_function ([some_digit ])# returns a score for each instance
threshold = 0
y_some_digit_pred = (y_scores > threshold )
# raise threshold
#raising the threshold decreases recall because of all instances
# that are true in reality, less will be classified as true
threshold = 3000
y_some_digit_pred = (y_scores > threshold )
y_some_digit_pred

# decide which threshold to use :
y_scores = cross_val_predict (sgd_clf , X_train , y_train_5 , cv=3, method ="decision_function" )
#With these scores, use theprecision_recall_curve()
# to compute precision and recall for all possible thresholds
# (the function adds a last precision of 0 and a last recall of 1,
# corresponding to an infinite threshold
from sklearn.metrics import precision_recall_curve
precisions , recalls , thresholds = precision_recall_curve (y_train_5 , y_scores )


# argmax returns the first index of the maximum value,
# which in this case means the first True value
idx_for_90_precision = (precisions >= 0.90 ).argmax ()
threshold_for_90_precision = thresholds [idx_for_90_precision ]
threshold_for_90_precision
# use this threshold 
y_train_pred_90 = (y_scores >= threshold_for_90_precision )
# check precision and recall
precision_score (y_train_5 , y_train_pred_90 )
recall_at_90_precision = recall_score (y_train_5 , y_train_pred_90 ) 
recall_at_90_precision

# false positive rate = FPR or fall out     : ratio of negative instances that are incorrectly classified as positive.
# true negative rate = TNR ( specificity)   : ratio of negative instances that are correctly classified as negative 
# ROC curve plots sensitivity (recall) versus 1 – specificity .

from sklearn.metrics import roc_curve
fpr, tpr , thresholds = roc_curve (y_train_5 , y_scores)
#you should prefer the PR curve (precision,recall) whenever the positive class
# is rare or when you care more about the false positives than the false negatives
# plot in main

#One way to compare classifiers is to measure the area under the curve (AUC)
#A perfect classifier will have a ROC AUC equal to 1,
# whereas a purely random classifier will have a ROC AUC equal to 0.5
from sklearn.metrics import roc_auc_score
roc_auc_score (y_train_5 , y_scores )

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier (random_state =42)

y_probas_forest = cross_val_predict (forest_clf , X_train , y_train_5 , cv=3, method ="predict_proba" )
y_probas_forest [:2] # array([[0.11, 0.89],
                        #     [0.99, 0.01]]) either positive or negative so rows add up to 1

#he second column contains the estimated probabilities for the positive class [:, 1]
y_scores_forest = y_probas_forest [:, 1]
precisions_forest , recalls_forest , thresholds_forest = precision_recall_curve ( y_train_5 , y_scores_forest )
# plot in main

y_train_pred_forest = y_probas_forest [:, 1] >= 0.5
f1_rf=f1_score (y_train_5 , y_train_pred_forest )
print(f1_rf)

roc_auc_score (y_train_5 , y_scores_forest )

if __name__=="__main__":
    # doesnt work
    #plt.plot (thresholds , precisions [:-1], "b--" , label ="Precision" , linewidth =2)
    #plt.plot (thresholds , recalls [:-1], "g-" , label ="Recall" , linewidth =2)
    #plt.vlines (threshold , 0, 1.0, "k" , "dotted" , label ="threshold" )
    #[... ] # beautify the figure: add grid, legend, axis, labels, and circles
    #plt.show ()
    
    
    #plt.plot (recalls , precisions , linewidth =2, label ="Precision/Recall curve" )
    #[... ] # beautify the figure: add labels, grid, legend, arrow, and text
    #plt.show ()
    
    #idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision ).argmax ()
    #tpr_90 , fpr_90 = tpr [idx_for_threshold_at_90 ], fpr[idx_for_threshold_at_90 ]
    #plt.plot (fpr, tpr , linewidth =2, label ="ROC curve" )
    #plt.plot ([0, 1], [0, 1], 'k:' , label ="Random classifier's ROC curve" )
    #plt.plot ([fpr_90 ], [tpr_90 ], "ko" , label ="Threshold for 90% precision" )
    #[... ] # beautify the figure: add labels, grid, legend, arrow, and text
    #plt.show ()
    
    #rf 
    plt.plot (recalls_forest , precisions_forest , "b-" , linewidth =2, label ="Random Forest" )
    plt.plot (recalls , precisions , "--" , linewidth =2, label ="SGD" )
    #[... ] # beautify the figure: add labels, grid, and legend
    plt.show 