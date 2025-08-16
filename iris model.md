from typing\_extensions import Counter

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn import datasets

from sklearn.model\_selection import train\_test\_split

from sklearn.preprocessing import Normalizer

from sklearn.metrics import accuracy\_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model\_selection import KFold

from collections import Counter











\#  import iris dataset

iris = datasets.load\_iris()

\# np.c is the nump concatate fuction

iris\_df = pd.DataFrame(data=np.c\_\[iris\['data'], iris\['target']],

&nbsp;                      columns=iris\['feature\_names'] + \['target'])







iris\_df.head ()

iris\_df.describe ()























\# split into x and y

X = iris\_df.iloc\[:, :-1]

y = iris\_df.iloc\[:, -1]



X.head()











y.head()



















\# split data into training and test sets



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&nbsp;   X, y, test\_size=0.2, shuffle=True, random\_state=1

)





X\_train = np.asarray(X\_train)

y\_train = np.asarray(y\_train)

X\_test = np.asarray(X\_test)

y\_test = np.asarray(y\_test)









print (f"trainning test size" \[X\_train.shape\[0]) samples n /test set size: X+test\[shape\[0]) samples"







\#normalize the dataset 



scaler = Normalizer().fit(X\_train)   # creates  normalizer and fit to the training data

normalized\_X\_train = scaler.transform(X\_train) #applies normalization to training set

normalized\_X\_test = scaler.transform(X\_test) #  applies the same method to test set















print("X\_train before normalization")



print(X\_train\[:5])



print("nx train after normalization")

print(normalized\_X\_train\[:5])

























\## Before

\# view the relationships between variables: color code by species type



d1 = {0.0: "setosa", 1.0: "versicolor", 2.0: "virginica"}







before = sns.pairplot(iris\_df.replace({"target": d1}), hue="target")

before.fig.suptitle("Pair plot of the dataset before normalization", y=1.02)





\## after



iris\_df\_2 = pd.dataframe(data=np.c\[normalized\_X\_train, y\_train]

columns = iris\["feature name"] + \["target"])





d1 = {0.0: "setosa", 1.0: "versicolor", 2.0: "virginica"}



after = sns.pairplot(iris\_df\_2.replace({"target": d1}), hue="target")

after.fig.suptitle("Pair plot of the dataset after normalization", y=1.02)



