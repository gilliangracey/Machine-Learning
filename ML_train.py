

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt




#Load data
feature_vector = pd.read_csv("feature_vector_verified_newstops.csv")[["asin", "Avg Summary Sentiment Score", 
                                                  "Avg Review Sentiment Score"]]
products = pd.read_csv("Toys_and_Gamesproduct_training.csv")

#combined feature vector with awesomeness score
feature_vector = feature_vector.merge(products, on="asin")

#model scores
model_scores = {}
model_precision = {}
model_recall = {}


"""
K-Nearest Neighbors:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
""" 
from sklearn.neighbors import KNeighborsClassifier
name = "K-Nearest Neighbors"
clf = KNeighborsClassifier()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")
precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""Guassian Naive Bayes: 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
"""
from sklearn.naive_bayes import GaussianNB
name = "Guassian Naive Bayes"
clf = GaussianNB()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""Decision Trees:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
"""
from sklearn.tree import DecisionTreeClassifier
name = "Decision Trees"
clf = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""SVC:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""
from sklearn.svm import SVC
name = "SVC"
clf = SVC()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""Logistic Regression:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
"""
from sklearn.linear_model import LogisticRegression
name = "Logistic Regression"
clf = LogisticRegression()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""Random Forest:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
"""
from sklearn.ensemble import RandomForestClassifier
name = "Random Forest"
clf = RandomForestClassifier()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


"""Gradient Boosting:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
"""
from sklearn.ensemble import GradientBoostingClassifier
name = "Gradient Boosting"
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")

precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall



"""Hist Gradient Boosting (faster):
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
"""
from sklearn.ensemble import HistGradientBoostingClassifier
name = "Hist Gradient Boosting"
clf = HistGradientBoostingClassifier()
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")
precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall



"""AdaBoost:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
"""
from sklearn.ensemble import AdaBoostClassifier
name = "AdaBoost"
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="f1")
precision = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="precision")
recall = cross_val_score(clf, 
                        feature_vector[["Avg Summary Sentiment Score","Avg Review Sentiment Score"]], 
                        feature_vector["awesomeness"],
                        cv=10, scoring="recall")

model_scores[name] = scores
model_precision[name] = precision
model_recall[name] = recall


mean = []
std_dev = []
labels = []

for key in model_scores:
    labels.append(key)
    mean.append(model_scores[key].mean())
    std_dev.append(model_scores[key].std())
    

x_pos = np.arange(len(labels))
fig, ax1 = plt.subplots(figsize=(10, 10))

for i in range(len(labels)):
    ax1.bar(i, mean[i], align='center', color = 'royalblue', alpha=1,zorder=3)

ax1.set(
      title='Comparison of Classifier Models: F1',
      ylabel='Mean F1 Score for 10-fold Cross Validation')

ax1.yaxis.grid(True, linestyle='-', which='major', color='black',
               alpha=0.5, zorder=0)

plt.xticks(x_pos, labels, rotation= 90)
plt.ylim([0,1])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("Accuracy Comparisons.png")

mean = []
std_dev = []
labels = []

for key in model_precision:
    labels.append(key)
    mean.append(model_precision[key].mean())
    std_dev.append(model_precision[key].std())
    

x_pos = np.arange(len(labels))
fig, ax1 = plt.subplots(figsize=(10, 10))

for i in range(len(labels)):
    ax1.bar(i, mean[i], align='center', color = 'royalblue', alpha=1,zorder=3)

ax1.set(
      title='Comparison of Classifier Models: Precision',
      ylabel='Mean Precision Score for 10-fold Cross Validation')

ax1.yaxis.grid(True, linestyle='-', which='major', color='black',
               alpha=0.5, zorder=0)

plt.xticks(x_pos, labels, rotation= 90)
plt.ylim([0,1])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("Precision Comparisons.png")

mean = []
std_dev = []
labels = []

for key in model_recall:
    labels.append(key)
    mean.append(model_recall[key].mean())
    std_dev.append(model_recall[key].std())
    

x_pos = np.arange(len(labels))
fig, ax1 = plt.subplots(figsize=(10, 10))

for i in range(len(labels)):
    ax1.bar(i, mean[i], align='center', color = 'royalblue', alpha=1,zorder=3)

ax1.set(
      title='Comparison of Classifier Models: Recall',
      ylabel='Mean Recall Score for 10-fold Cross Validation')

ax1.yaxis.grid(True, linestyle='-', which='major', color='black',
               alpha=0.5, zorder=0)

plt.xticks(x_pos, labels, rotation= 90)
plt.ylim([0,1])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.savefig("Recall Comparisons.png")


#print(model_scores)
#print(model_precision)
#print(model_recall)

