from sklearn.model_selection import cross_val_score
import time as t
import numpy as np
from sklearn import metrics
import pandas as pd
import sklearn.metrics as metrics
from skimage.restoration import estimate_sigma
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import csv
import statistics as stat
start = t.time()
"""
red = pd.read_csv('/home/lenovo/Python3/RED.csv')
blue = pd.read_csv('/home/lenovo/Python3/BLUE.csv')
green = pd.read_csv('/home/lenovo/Python3/GREEN.csv')
"""
F = pd.read_csv('/home/lenovo/Python3/Features.csv')
A = F.iloc[:,:-1].values
b = F.iloc[:,7].values
X = A
y = b
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.25)
scaler = StandardScaler()
scaler.fit(A_train)
A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
k_range = range(1,51)
scores = {}
time = []
scores_list = []
for k in k_range:
	s = t.time()
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(A_train,b_train)
	b_pred=knn.predict(A_test)
	with open('K.csv','a') as f:
		c = csv.writer(f)
		c.writerow(str(metrics.accuracy_score(b_test,b_pred)))
		scores_list.append(metrics.accuracy_score(b_test,b_pred))
	e = t.time()
	time.append(e-s)
x = []
ind = 11
f1 = plt.figure(1)
plt.title('Distribution For K')
plt.plot(k_range,scores_list,label = "K value Distribution",marker = 'o')
plt.legend(loc = "lower right")
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.25)
scaler = StandardScaler()
scaler.fit(A_train)
A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
classifier = KNeighborsClassifier(n_neighbors=ind)
classifier.fit(A_train, b_train)
b_pred = classifier.predict(A_test)
#loo = LeaveOneOut()
knn_cv = KNeighborsClassifier(n_neighbors=ind)
cv_scores = cross_val_score(knn_cv, X, y, cv=7,n_jobs = -1)
fields = ["                     ",'Actual NO','Actual Yes']
rows = [['Predicted No',confusion_matrix(b_test, b_pred)[0][0],confusion_matrix(b_test, b_pred)[0][1]],['Predicted Yes',confusion_matrix(b_test, b_pred)[1][0],confusion_matrix(b_test, b_pred)[1][1]]]
with open('Report.csv', 'w') as f:
	f.write("\nConfusion Matrix\n")
	csvwriter = csv.writer(f)
	csvwriter.writerow(fields)  
	csvwriter.writerows(rows)
	f.write("\nClassification Report \n")
	f.write(str(classification_report(b_test, b_pred)))
	f.write("\nCross Value Score\n")
	f.write(str(cv_scores))
	f.write("\n Mean Cross Value Score \n ")
	f.write(str(np.mean(cv_scores)))
	f.write("\nAccuracy Applying Normal KNN\n")
	x = "%"
	f.write(str(classifier.score(A_test,b_test))+x)
	f.write("\nAccuracy Applying Cross Validation\n")
	f.write(str(np.mean(cv_scores))+x)
model = LogisticRegression(solver='lbfgs')
model.fit(A,b)
probs = model.predict_proba(A_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(b_test, preds)
roc_auc = metrics.auc(fpr, tpr)
f2 = plt.figure(2)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc,color = 'red',marker = 'o')
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
"""
A = red.iloc[:,:-1].values
b = red.iloc[:,7].values
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.20)
scaler = StandardScaler()
scaler.fit(A_train)
A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(A_train, b_train)
b_pred = classifier.predict(A_test)
fields = ['BLUE','Actual NO','Actual Yes']
rows = [['Predicted No',confusion_matrix(b_test, b_pred)[0][0],confusion_matrix(b_test, b_pred)[0][1]],['Predicted Yes',confusion_matrix(b_test, b_pred)[1][0],confusion_matrix(b_test, b_pred)[1][1]]]
with open("CM.csv", 'a') as csvfile:  
    	csvwriter = csv.writer(csvfile)          
    	csvwriter.writerow(fields)  
    	csvwriter.writerows(rows)
with open('CR.csv', 'a') as f:
	f.write("RED")
	f.write(str(classification_report(b_test, b_pred)))
model = LogisticRegression(solver='lbfgs')
model.fit(A,b)
probs = model.predict_proba(A_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(b_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc,color = 'blue')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
A = blue.iloc[:,:-1].values
b = blue.iloc[:,7].values
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.20)
scaler = StandardScaler()
scaler.fit(A_train)
A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(A_train, b_train)
b_pred = classifier.predict(A_test)
fields = ['BLUE','Actual NO','Actual Yes']
rows = [['Predicted No',confusion_matrix(b_test, b_pred)[0][0],confusion_matrix(b_test, b_pred)[0][1]],['Predicted Yes',confusion_matrix(b_test, b_pred)[1][0],confusion_matrix(b_test, b_pred)[1][1]]]
with open("CM.csv", 'a') as csvfile:  
    	csvwriter = csv.writer(csvfile)          
    	csvwriter.writerow(fields)  
    	csvwriter.writerows(rows)
with open('CR.csv', 'a') as f:
	f.write("BLUE")
	f.write(str(classification_report(b_test, b_pred)))
model = LogisticRegression(solver='lbfgs')
model.fit(A,b)
probs = model.predict_proba(A_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(b_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc,color = 'blue')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
A = green.iloc[:,:-1].values
b = green.iloc[:,7].values
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.5)
scaler = StandardScaler()
scaler.fit(A_train)
A_train = scaler.transform(A_train)
A_test = scaler.transform(A_test)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(A_train, b_train)
b_pred = classifier.predict(A_test)
fields = ['GREEN','Actual NO','Actual Yes']
rows = [['Predicted No',confusion_matrix(b_test, b_pred)[0][0],confusion_matrix(b_test, b_pred)[0][1]],['Predicted Yes',confusion_matrix(b_test, b_pred)[1][0],confusion_matrix(b_test, b_pred)[1][1]]]
with open("CM.csv", 'a') as csvfile:  
    	csvwriter = csv.writer(csvfile)          
    	csvwriter.writerow(fields)  
    	csvwriter.writerows(rows)
with open('CR.csv', 'a') as f:
	f.write("GREEN")
	f.write(str(classification_report(b_test, b_pred)))
model = LogisticRegression(solver='lbfgs')
model.fit(A,b)
probs = model.predict_proba(A_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(b_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc,color = 'green')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
"""
end = t.time()
print(end - start)
plt.show()
