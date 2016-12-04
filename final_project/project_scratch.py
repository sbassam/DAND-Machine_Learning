#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot


###########################FEATURE SELECTION##########################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print 'total number of data points:	', len(data_dict)
print 'total number of data points labeled as POI:	', sum(1 for i in data_dict if data_dict[i]['poi'] == 1)
print 'total number of data points labeled as non-POI:	', sum(1 for i in data_dict if data_dict[i]['poi'] == 0)
print 'total number of features:	', len(data_dict.values()[1])
print 'count of data point with NaN director_fees:	', sum (1 for i in data_dict if data_dict[i]['director_fees'] == 'NaN')

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
	salary	= point[0]
	bonus	= point[1]
matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Task 2: Remove outliers

#print data_dict.keys()
data_dict.pop( 'TOTAL', 0 )



### your code below


#features = ["poi", "from_messages", 'from_this_person_to_poi', "salary"]
features = ["poi", "from_messages", 'from_this_person_to_poi', "salary"]
data = featureFormat(data_dict, features)

for point in data:
	poi = point[0]
	proportion = point[2]/point[1]
	#from_messages = point [1]
	#from_this_person_to_poi = point[2]
	if point[0] == 1:
		matplotlib.pyplot.scatter( proportion, poi, color = "r" )
	else:
		matplotlib.pyplot.scatter( proportion, poi, color = "b" )

matplotlib.pyplot.xlabel("proportion")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()


###############################OUTLIERS#############################
data_dict.pop( 'TOTAL', 0 )
# removing TOTAL resultS in the following:
# scores: huge increase
# precision and recall: slight decrease

#############################NEW FEATURES#############################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for i in data_dict:
	if (data_dict[i]["from_this_person_to_poi"] == "NaN" or data_dict[i]["from_messages"] == "NaN" or data_dict[i]["from_messages"] == 0):
		data_dict[i]['poi_to_total_proportion'] = 0
	else:
		data_dict[i]['poi_to_total_proportion'] = float(data_dict[i]["from_this_person_to_poi"])/float(data_dict[i]["from_messages"])
	#poi_to_total_proportion = v["from_this_person_to_poi"]/v["from_messages"]
#print data_dict


my_dataset = data_dict
print my_dataset['METTS MARK'].keys()



### Extract features and labels from dataset for local testing
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
features_train2 = sel.fit(features, labels)
feature_scores = sel.scores_
feature_pvalues = sel.pvalues_
feature_indices = sel.get_support(indices=True)
features_scores_values = [(features_list[i+1], feature_scores[i], feature_pvalues[i]) for i in feature_indices]
from operator import itemgetter
features_scores_values = sorted(features_scores_values,key=itemgetter(1))
#features_test2  = sel.transform(features_test)#.toarray()
#print features_scores_values

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

features_list = [
	'poi',
	'poi_to_total_proportion',
	#'exercised_stock_options',
	'total_stock_value',
	'bonus',
	'salary',
	'deferred_income',
	'long_term_incentive',
	'restricted_stock',
	'total_payments',
	'shared_receipt_with_poi',
	'loan_advances',
	'expenses',
	'from_poi_to_this_person',
	#'from_this_person_to_poi',
	'director_fees'
	]
features_list = [
	'poi',
	'poi_to_total_proportion',
	#'exercised_stock_options',
	'total_stock_value',
	'bonus',
	#'salary',
	'deferred_income',
	#'long_term_incentive',
	'restricted_stock',
	'total_payments',
	#'shared_receipt_with_poi',
	#'loan_advances',
	'expenses',
	#'from_messages',
	'from_poi_to_this_person'
	#'from_this_person_to_poi',
	#'director_fees'
	]



data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


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

'''from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_train = min_max_scaler.fit_transform(features_train)'''
#features_train = preprocessing.scale(features_train)
'''from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
nb_score = clf.score(features_test, labels_test)
print 'naive bayes score: ', nb_score
# ok! 0.34 is pretty low, but at least i got something up & running.'''
print '####################################TREE##########################################'
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state = 19, min_samples_split = 2, max_leaf_nodes = None, criterion = 'entropy', max_depth = 2, min_samples_leaf =1)
# TUNED TREE: tree.DecisionTreeClassifier(random_state = 19, min_samples_split = 2, max_leaf_nodes = 5, criterion = 'entropy', max_depth = None, min_samples_leaf =1) recall: 0.17 precision: .36
#			Also: , min_samples_split = 4, max_leaf_nodes = 9, criterion = 'entropy', max_depth = None, min_samples_leaf =1 HIGHEST P: .31 R: .21 F1: .25
# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
tree_score = clf.score(features_test, labels_test)
print  'accuracy score for tree: ', tree_score

# whoa! almost double. time to implement more evaluation metrics

tree_precision = precision_score(labels_test, pred)
print 'precision for tree: ', tree_precision
tree_recall = recall_score(labels_test, pred)
print 'recall for tree: ', tree_recall
###USE THIS clf.feature_importances_
# good! already higher than the required .3 precision. let's see what can improve it.

print '#####################################TUNED TREE#################################'
'''from sklearn import svm, datasets
from sklearn.grid_search import GridSearchCV
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2]}
parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [1, 2, 4],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 2, 5, 9],

              }
svr = tree.DecisionTreeClassifier()
#svr = svm.SVC()
print 'a'
clf = GridSearchCV(svr, parameters)
print 'b'
clf.fit(features, labels)
print 'c'
#result = sorted(clf.cv_results_.keys())
#print result
print clf.best_params_'''
print '##############################KNN########################################'
'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform', algorithm = 'auto')
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
print 'recall for knn: ', knn_recall'''
#import matplotlib.pyplot as plt
#from class_vis import prettyPicture

#try:
#    prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass


'''from sklearn.grid_search import GridSearchCV
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2]}
parameters = {"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
              "n_neighbors": [3, 5, 7, 10],
              "weights": ["uniform", "distance"]
              }
clf = KNeighborsClassifier()
#svr = svm.SVC()
print 'a'
clf = GridSearchCV(clf, parameters)
print 'b'
clf.fit(features, labels)
print 'c'
#result = sorted(clf.cv_results_.keys())
#print result
print clf.best_params_'''
############################SVM########################################
'''from sklearn.svm import SVC
clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

# fit:
clf.fit(features_train, labels_train)

# predict:
pred = clf.predict(features_test)

# score:
svm_score = clf.score(features_test, labels_test)
print svm_score

# whoa! almost double. time to implement more evaluation metrics
svm_precision = precision_score(labels_test, pred)
print 'precision for svm: ', svm_precision
svm_recall = recall_score(labels_test, pred)
print 'recall for svm: ', svm_recall

'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from sklearn.naive_bayes import GaussianNB
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

for param_name in sorted(parameters.keys()):
        print("\t{}: {}".format(param_name, best_parameters[param_name]))
print 'finish'



###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
n_components = 20


pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)



X_train_pca = pca.transform(features_train)
X_test_pca = pca.transform(features_test)

print 'explained variances: ', sorted(zip(features_list[1:], pca.explained_variance_ratio_),key=itemgetter(1))


###############################################################################
'''# Train a SVM classification model
print "Fitting the classifier to the training set"

param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, labels_train)

print "Best estimator found by grid search:"
print clf.best_estimator_

'''

dump_classifier_and_data(clf, my_dataset, features_list)

#source: http://abshinn.github.io/python/sklearn/2014/06/08/grid-searching-in-all-the-right-places/