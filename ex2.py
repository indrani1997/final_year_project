from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt      
import pandas as pd 
import numpy as np
import mglearn
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv('wisconsin_data.csv')
z=data.loc[:,"nucleoli"].mean()
z=int(z)
x=data.iloc[:,1:].values
#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
#print (imputer.fit(x[:,6:7]))
#x[:,6:7]=imputer.transform(x[:,6:7])
x=data.iloc[0:,1:10].values
y=data.iloc[0:,10].values

for i in x:
    if str(i[5])=="nan":
        i[5]=z

#print ("this arethe values stored in columns",x[1:,6])
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.3,random_state=1)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
print ('Accuracy of KNN n=5 ,on the training set : {0}'.format(knn.score(X_train,Y_train)))
print("accuracy of Knn n=5 on the test set {0}".format(knn.score(X_test,Y_test)))

dtc=DecisionTreeClassifier(random_state=1)
dtc.fit(X_train,Y_train)

y_pred_class=dtc.predict(X_test)
accuracy=metrics.accuracy_score(Y_test,y_pred_class)
print("accuracy of Decison Tree is {0}",accuracy)