import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df = pd.read_csv('C:/Users/Lenovo/Downloads/Covid Dataset.csv')
df.replace(to_replace = ['Yes','No'],
                 value =[1,0],inplace = True)
cor_M = df.corr()

### Pre -Processing

Uzlessfeatures = cor_M[(cor_M['COVID-19'] <= 0) | (cor_M['COVID-19'].isnull())]
ind = Uzlessfeatures.index
df = df.drop(ind,axis=1)
X = df.iloc[:,0:13]
y = df.iloc[:,13]
sm = SMOTE()
X,y = sm.fit_resample(X,y)

## Model Building
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

pickle.dump(svclassifier, open('model.pkl', 'wb'))

y_predg = svclassifier.predict(X_test)