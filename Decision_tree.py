import numpy as np
import pandas as pd

#reading dataset
dataset=pd.read_csv("DT_dataset.csv")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4:] #last column

#Perform Label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()
X = X.apply(labelencoder_X.fit_transform) #fit_transform - Fit to data, then transform it into numerical values, transform- Perform standardization by centering and scaling
print(X)
y = y.apply(labelencoder_Y.fit_transform)

print(y)

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()  #by default gini indes is used for selecting the root node
regressor.fit(X.iloc[:,:],y)    #training the model. 

X_in=["21","Low","Female","Married"]
X_inn=labelencoder_X.fit_transform(X_in)
y_pred = regressor.predict([X_inn])

print("Prediction for : ",X_in )
print("Prediction :", labelencoder_Y.inverse_transform(y_pred))

#from io import StringIO
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus

#dot_data = StringIO()

#export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_character=True)
#graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png("tree.png")




