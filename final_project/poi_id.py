#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot
from operator import itemgetter


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


print '#########################REMOVE OUTLIERS#########################'

# plot to find out if there are outliers
features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.suptitle("Before Removing the Outlier")
matplotlib.pyplot.show()

# Remove the outlier:
data_dict.pop( 'TOTAL', 0 )

# plot again:
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.suptitle("After Removing the Outlier")
matplotlib.pyplot.show()

print '##################################################'

print '#########################FEATURE SELECTION#########################'
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Here's the list from which I'm going to pick features. 
features_list = [
		'poi',
		'salary', 
		'to_messages',
		'deferral_payments',
		'total_payments', 
		'exercised_stock_options',
		'bonus',
		'restricted_stock',
		'shared_receipt_with_poi',
		'restricted_stock_deferred',
		'total_stock_value', 
		'expenses', 
		'loan_advances', 
		'from_messages', 
		'from_this_person_to_poi', 
		'director_fees', 
		'deferred_income', 
		'long_term_incentive', 
		'from_poi_to_this_person'
	]

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
sel = SelectKBest(k = 'all')
sel.fit(features, labels)
feature_scores = sorted(zip(features_list[1:], sel.scores_), key = itemgetter(1))
print 'SelectKBest feature scores:	', feature_scores
# Initially, I chose the features based on the SelectKBest scores, but since I was getting terrible Recalls, I went back and added 
# 		feature_importances_ attribute to the decision tree and chose features based on it. 

# creating a new feature

features_plot = ["poi", "from_messages", 'from_this_person_to_poi']
data = featureFormat(data_dict, features_plot)

# poi vs from_this_person_to_poi/from_messages plot:
for point in data:
	poi = point[0]
	proportion = point[2]/point[1]
	if point[0] == 1:
		matplotlib.pyplot.scatter( proportion, poi, color = "r" )
	else:
		matplotlib.pyplot.scatter( proportion, poi, color = "b" )
matplotlib.pyplot.xlabel("proportion")
matplotlib.pyplot.ylabel("poi")
matplotlib.pyplot.suptitle("poi vs from_this_person_to_poi/from_messages")
matplotlib.pyplot.show()

# from_messages vs from_this_person_to_poi plot:
for point in data:
	from_messages = point [1]
	from_this_person_to_poi = point[2]
	if point[0] == 1:
		matplotlib.pyplot.scatter( from_this_person_to_poi, from_messages, color = "r" )
	else:
		matplotlib.pyplot.scatter( from_this_person_to_poi, from_messages, color = "b" )

matplotlib.pyplot.xlabel("from_this_person_to_poi")
matplotlib.pyplot.ylabel("from_messages")
matplotlib.pyplot.suptitle("poi vs from_this_person_to_poi/from_messages")
matplotlib.pyplot.show()

# Add the new feature poi_to_total_proportion:

for i in data_dict:
	if (data_dict[i]["from_this_person_to_poi"] == "NaN" or data_dict[i]["from_messages"] == "NaN" or data_dict[i]["from_messages"] == 0):
		data_dict[i]['poi_to_total_proportion'] = 0
	else:
		data_dict[i]['poi_to_total_proportion'] = float(data_dict[i]["from_this_person_to_poi"])/float(data_dict[i]["from_messages"])

my_dataset = data_dict

# new features_list:

features_list = [
	'poi',
	'poi_to_total_proportion',
	#'exercised_stock_options', REVISED
	'total_stock_value',
	'bonus',
	#'salary', REVISED
	'deferred_income',
	#'long_term_incentive', REVISED
	'restricted_stock',
	'total_payments',
	#'shared_receipt_with_poi', REVISED
	#'loan_advances', REVISED
	'expenses',
	'from_poi_to_this_person'
	#'from_this_person_to_poi', USED THE NEW FEATURE INSTEAD
	#'director_fees' REVISED
	]

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.'''

print '#########################DECISION TREE (OUT OF THE BOX)#########################'

clf = tree.DecisionTreeClassifier(random_state = 19)

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
tree_score = clf.score(features_test, labels_test)
print  'accuracy score for tree: ', tree_score

tree_precision = precision_score(labels_test, pred)
print 'precision for tree: ', tree_precision
tree_recall = recall_score(labels_test, pred)
print 'recall for tree: ', tree_recall
importance = sorted(zip(features_list[1:],clf.feature_importances_), key=itemgetter(1))
print 'feature importances:	', importance

print '################################################################################'

print '##############################KNN########################################'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf = KNeighborsClassifier(n_neighbors = 2)
clf.fit(features_train, labels_train) 

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print 'accuracy', accuracy
# score:
knn_score = clf.score(features_test, labels_test)
print  'accuracy score for knn: ', knn_score

print labels_test
print pred
knn_precision = precision_score(labels_test, pred)
print 'precision for knn: ', knn_precision
knn_recall = recall_score(labels_test, pred)
print 'recall for knn: ', knn_recall

print '################################################################################'

print '#################################Naive Base#####################################'
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
nb_score = clf.score(features_test, labels_test)
print 'naive bayes score: ', nb_score
print '################################################################################'

print '#################################SVM#####################################'
from sklearn.svm import SVC
clf = SVC()

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
svm_score = clf.score(features_test, labels_test)
print svm_score
print '################################################################################'


print '################################GridSearchCV####################################'
'''
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

pipeline = Pipeline([
    ( 'tree', tree.DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
])

parameters = {
    #'knn__C': (1, 10),
    #'tree__penalty': ('l1', 'l2'),
    'knn__n_neighbors': (2, 5, 10),
    'tree__min_samples_split': (1,2,4),
    #'tree__min_samples_split' :(2, 3, 4), 
    'tree__max_leaf_nodes' :(None, 2, 4,), 
    'tree__criterion': ('entropy', 'gini'), 
    'tree__max_depth': (2, 4, 6), 
    'tree__min_samples_leaf':(1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters,scoring = "roc_auc")

grid_search.fit(features_train, labels_train)

best_parameters = grid_search.best_estimator_.get_params()

for i in sorted(parameters.keys()):
        print("\t{}: {}".format(i, best_parameters[i]))
'''
print '################################################################################'

print '######################################KNN TUNED##########################################'
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(features_train, labels_train) 

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print 'accuracy', accuracy
# score:
knn_score = clf.score(features_test, labels_test)
print  'accuracy score for knn: ', knn_score

print labels_test
print pred
knn_precision = precision_score(labels_test, pred)
print 'precision for knn: ', knn_precision
knn_recall = recall_score(labels_test, pred)
print 'recall for knn: ', knn_recall
print '################################################################################'

print '######################################DECISION TREE TUNED##########################################'

clf = tree.DecisionTreeClassifier(random_state = 19, min_samples_split = 2, 
	max_leaf_nodes = None, criterion = 'entropy', max_depth = 2, min_samples_leaf =1)

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
tree_score = clf.score(features_test, labels_test)
print  'accuracy score for tree: ', tree_score


tree_precision = precision_score(labels_test, pred)
print 'precision for tree: ', tree_precision
tree_recall = recall_score(labels_test, pred)
print 'recall for tree: ', tree_recall
importance = sorted(zip(features_list[1:],clf.feature_importances_), key=itemgetter(1))
print 'feature importances:	', importance

print '################################################################################'

print '######################################PCA##########################################'
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
n_components = 8


pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)



X_train_pca = pca.transform(features_train)
X_test_pca = pca.transform(features_test)

print 'explained variances: ', sorted(zip(features_list[1:], pca.explained_variance_ratio_),key=itemgetter(1))

print '################################################################################'


dump_classifier_and_data(clf, my_dataset, features_list)