
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('explore different classifier')

dataset_name = st.sidebar.selectbox("Select Dataset",("iris dataset","breast cancer","wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest "))

def get_dataset(dataset_name):
    if dataset_name == "iris dataset":
        datas = datasets.load_iris()
    elif dataset_name == "breast cancer":
        datas = datasets.load_breast_cancer()
    else:
        datas = datasets.load_wine()
    x = datas.data
    y = datas.target
    return x,y

x,y =get_dataset(dataset_name)

st.write("shape of dataset ",x.shape)
st.write("number of classes",len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        k = st.sidebar.slider("K",1,15)
        params["k"] = k
    elif clf_name =="SVM":
        c = st.sidebar.slider("C",.01,10.0)
        params["c"] = c
    else :
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimator",1,100)
        params["max_depth"]= max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["k"])
    elif clf_name =="SVM":
        clf = SVC(C = params["c"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"],
                                    random_state=1)
    return clf

clf = get_classifier(classifier_name,params)

#classification
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.2,random_state =1)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)
st.write("classifier = ",classifier_name)
st.write("accuracy = ",acc)

#plot

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha = 0.8 ,cmap ="viridis")
plt.xlabel("PRINCIPAL COMPONENT 1")
plt.ylabel("PRINCIPAL COMPONENT 2")

st.pyplot()
