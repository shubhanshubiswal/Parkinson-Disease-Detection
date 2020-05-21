#Import_Library
import numpy as np
import pandas as pd
import pickle

#Import_Dataset
dataset = pd.read_csv('Parkinson.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Splitting_Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Fitting XGBoost model to training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = 50, gamma = 0.1, max_depth = 5)
classifier.fit(X_train, y_train)

#predicting
y_pred = classifier.predict(X_test)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
arr = np.array([[119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]])
print(model.predict(arr))
print(model.predict_proba(arr))