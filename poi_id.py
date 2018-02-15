#!/usr/bin/python
from __future__ import division
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use. 
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["salary", "bonus"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Data description
# print "No of data Points: ",len(data_dict)
# print "No of Features: ",max(len(data_dict[i]) for i in data_dict)
# print "No of POIs(Points of Intrest): ",len([1 for i in data_dict if data_dict[i]['poi'] == 1])
# print "No of Non POIs: ",len([1 for i in data_dict if data_dict[i]['poi'] != 1])
# print "No of employees without salary: ", len([1 for i in data_dict if data_dict[i]['salary'] == 'NaN'])
# print "No of employees with salary: ", len([1 for i in data_dict if data_dict[i]['salary'] != 'NaN'])
# print "No of employees with unknown total payments: ", len([1 for i in data_dict if data_dict[i]['total_payments'] == 'NaN'])
# print "No of POI where total payment is NaN: ", len([1 for j in [data_dict[i]['total_payments'] for i in data_dict if data_dict[i]['poi'] == 1] if j == 'NaN'])
# print "No of employees with email address: ",len([1 for i in data_dict if data_dict[i]['email_address'] != 'NaN'])

# # missing values for all
# print "|Missing deferral payments|", len([1 for i in data_dict if data_dict[i]["deferral_payments"] == 'NaN'])*100/len(data_dict) 
# print "|Missing loan advances|", len([1 for i in data_dict if data_dict[i]["loan_advances"] == 'NaN'])*100/len(data_dict) 
# print "|Missing restricted stock deferred|", len([1 for i in data_dict if data_dict[i]["restricted_stock_deferred"] == 'NaN'])*100/len(data_dict) 
# print "|Missing deferred income|", len([1 for i in data_dict if data_dict[i]["deferred_income"] == 'NaN'])*100/len(data_dict) 
# print "|Missing long term incentive|", len([1 for i in data_dict if data_dict[i]["long_term_incentive"] == 'NaN'])*100/len(data_dict) 
# print "|Missing director fees|", len([1 for i in data_dict if data_dict[i]["director_fees"] == 'NaN'])*100/len(data_dict) 

# list_of_poi = [data_dict[i] for i in data_dict if data_dict[i]['poi'] == 1]

# # Missing values For POIs
# print "|Missing deferral payments|", len([1 for i in list_of_poi if i["deferral_payments"] == 'NaN'])*100/len(list_of_poi) 
# print "|Missing loan advances|", len([1 for i in list_of_poi if i["loan_advances"] == 'NaN'])*100/len(list_of_poi) 
# print "|Missing restricted stock deferred|", len([1 for i in list_of_poi if i["restricted_stock_deferred"] == 'NaN'])*100/len(list_of_poi) 
# print "|Missing deferred income|", len([1 for i in list_of_poi if i["deferred_income"] == 'NaN'])*100/len(list_of_poi) 
# print "|Missing long term incentive|", len([1 for i in list_of_poi if i["long_term_incentive"] == 'NaN'])*100/len(list_of_poi) 
# print "|Missing director fees|", len([1 for i in list_of_poi if i["director_fees"] == 'NaN'])*100/len(list_of_poi) 

### plot features

### Task 2| Remove outliers

for outlier in ['TOTAL']:
    data_dict.pop(outlier,0)

### Task 3| Create new feature(s)

def get_ratio(num, denum):
	if num == 'NaN' or denum == 'NaN': return 0
	return num/denum

for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=get_ratio(data_dict[i]["from_poi_to_this_person"], data_dict[i]["to_messages"])
    data_dict[i]["fraction_to_poi_email"]=get_ratio(data_dict[i]["from_this_person_to_poi"],data_dict[i]["from_messages"])

### Store to my_dataset for easy export below.
my_dataset = data_dict

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"] 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# for point in data:
#     salary = point[0]
#     bonus = point[1]
#     plt.scatter( salary, bonus )
# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()

# for point in data:
#     from_poi = point[1]
#     to_poi = point[2]
#     plt.scatter( from_poi, to_poi )
#     if point[0] == 1:
#         plt.scatter(from_poi, to_poi, color="r", marker="*")
#     else:
#     	plt.scatter(from_poi, to_poi, color="b")
# plt.xlabel("fraction of emails persons gets from poi")
# plt.show()
 

#feature processing
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list)

#spliting the data into features and labels where labels is the status of a employee(is it a POI or not) 
labels, features = targetFeatureSplit(data)


from sklearn import cross_validation
#splitting the data in traning and test set
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

#importing decision tree classiifier
from sklearn.tree import DecisionTreeClassifier

# importing some formulas
from sklearn.metrics import precision_score, recall_score, accuracy_score

#importing time
from time import time


# Using decision tree
t0 = time()
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)

#calculating  precision, recll
precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)

print "INITIAL FINDING BEFORE ANY TUNING AND MACHINE LEARNING"
print "accuracy:%f"%score
print "precesion:%f"%precision
print "recall:%f"%recall
print "Decision tree algorithm time: %f%s"% (round(time()-t0, 3), "s")

print "******"

print "FEATURE IMPORTANCE LIST"

importance_list = clf.feature_importances_
import numpy as np
# importance_temp.sort(reverse=True)
# indices = [importances.index(i) for i in importance_temp]
indices = np.argsort(importance_list)[::-1]
print 'FEATURES  IMPORTANCE'
for i in range(16):
    print features_list[i+1],': ',importance_list[indices[i]]

print "******"
### Task 4: Try a varity of classifiers
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "shared_receipt_with_poi"]

### Please name your classifier clf for easy export below.


from sklearn.naive_bayes import GaussianNB


### Using Naive based
print 'FINDINGS USING NAIVE BAYES'
t0 = time()	
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)
print "Accracy: ",accuracy
print "precesion: ", precision
print "recall: ", recall

print "naive bayes algorithm time: %f%s"% (round(time()-t0, 3), "s")
print '********'

print 'FINDINGS USING DecisionTreeClassifier'
t0 = time()	
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)
print "Accracy: ",accuracy
print "precesion: ", precision
print "recall: ", recall

print "Decision Tree algorithm time: %f%s"% (round(time()-t0, 3), "s")
print '********'

print 'FINDINGS USING SVM'
from sklearn import svm
t0 = time()	
clf = svm.LinearSVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)
print "Accracy: ",accuracy
print "precesion: ", precision
print "recall: ", recall

print "SVM algorithm time: %f%s"% (round(time()-t0, 3), "s")
print '********'


### checking Decision tree with various minimum_split

print 'DECISION TREE ON VARIOUS SPLITS'
print "SPLIT\tACCURACY\tPRECISION\tRECALL\tTIME"
for i in range (2,10): 
    t0 = time()
    clf = DecisionTreeClassifier(min_samples_split=i)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    print "%d\t%0.4f\t\t%0.4f\t\t%0.4f\t%0.4f" % (i,accuracy,precision,recall,time()-t0)
print '********'

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']


### store to my_dataset for easy export below
my_dataset = data_dict

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)

# function uses Kfold algorithm to split data into traning and test set 
def KFold_split(labels, split_no = 2):
	from sklearn.cross_validation import KFold
	kf=KFold(len(labels),split_no)
	for train_indices, test_indices in kf:
	    features_train= [features[ii] for ii in train_indices]
	    features_test= [features[ii] for ii in test_indices]
	    labels_train=[labels[ii] for ii in train_indices]
	    labels_test=[labels[ii] for ii in test_indices]

	return features_train, features_test, labels_train, labels_test

#splitting data using kfold
features_train, features_test, labels_train, labels_test = KFold_split(labels,3)

### checking Decision tree with various minimum_split using kfold algorith
print 'DECISION TREE ON VARIOUS SPLITS USING KFOLD'
print "SPLIT\tACCURACY\tPRECISION\tRECALL\tTIME"
for i in range (2,10): 
    t0 = time()
    clf = DecisionTreeClassifier(min_samples_split=i)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    print "%d\t%0.4f\t\t%0.4f\t\t%0.4f\t%0.4f" % (i,accuracy,precision,recall,time()-t0)
print '********'

print 'FINAL RESULT AFTER ALL THE TUNING'
### use manual tuning with best parameters
t0 = time()
clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print "done in %0.3fs" % (time() - t0)

acc=accuracy_score(labels_test, pred)

print "accuracy after tuning = %f"% acc
print 'precision = %lf'% precision_score(labels_test,pred)
print 'recall = %lf'% recall_score(labels_test,pred)


### dump your classifier, dataset and features_list so
### anyone can run/check your results
dump_classifier_and_data(clf, my_dataset, features_list)




