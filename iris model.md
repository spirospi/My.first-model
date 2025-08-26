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







# knn algorithm

def distance_ecu(X_train, X_test_point):
    '''
      input:
               - X_train : corresponding to the training data
               - X_test_point : corresponding to the test point

      output:
              distances : the distances between the test point and each point in the training data
    '''

    distances = []  # store all distances here

    # loop over the rows of X_train, we calculate for each row in the training data
    for row in range(len(X_train)):
        current_train_point = X_train[row]   # get them point by point, for each row in X_train we go to the current row
        current_distance = 0                 # initialize the distance by zero

        # loop over the columns (features) of the row
        for col in range(len(current_train_point)):
            # add squared difference between the current feature of train point and test point
            current_distance += (current_train_point[col] - X_test_point[col]) ** 2

            # OR we could write:
            # current_distance = current_distance + (X_train[row][col] - X_test_point[col]) ** 2

        # take the square root to get Euclidean distance , we take the root from the distance which is not the distance and its the root
        current_distance = np.sqrt(current_distance)

        # append the distance to our list
        distances.append(current_distance)

    # store distances in a dataframe
    distances_df = pd.DataFrame(data=distances, columns=['dist'])

    return distances_df








def nearest_neighbors(distance_point, K):
    '''
    input :
             - distance_point : the distances between the test point and the training data
             - K              : the number of neighbors

    output:
             df_nearest : the nearest K neighbors between the test point and the training data
    '''

    # sort distances using the sort_values function
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    ## Take only the first k neighbors
    df_nearest = df_nearest[:K]

    return df_nearest






# knn step 3 (classify the point based on majority vote)

from collections import Counter   # make sure to import Counter

def voting(df_nearest, y_train):
    '''
    input
        - df_nearest : DataFrame containing the nearest k neighbors
                       between the full training data and the test point
        - y_train    : the labels of the training dataset

    output :
        y_pred : the prediction based on majority voting
    '''

    ## use the Counter object to get the labels of k nearest neighbors
    counter_vote = Counter(y_train.iloc[df_nearest.index])

    # majority voting: pick the most common label
    y_pred = counter_vote.most_common(1)[0][0]

    return y_pred













# -----------------------------
# 1. Distance function
# -----------------------------
def distance_ecu(X_train, X_test_point):
    distances = []
    for row in range(len(X_train)):
        current_train_point = X_train[row]
        current_distance = 0
        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - X_test_point[col]) ** 2
        current_distance = np.sqrt(current_distance)
        distances.append(current_distance)
    return distances

# -----------------------------
# 2. Nearest neighbor selector
# -----------------------------
def nearest_neighbor(distances, K):
    df = pd.DataFrame(data=distances, columns=['dist'])
    return df.sort_values(by='dist').head(K).index

# -----------------------------
# 3. Voting function
# -----------------------------
def voting(nearest_indices, y_train):
    nearest_labels = y_train[nearest_indices]
    vote_counts = Counter(nearest_labels)
    return vote_counts.most_common(1)[0][0]








from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=1
)

K = 3
y_pred = []

for X_test_point in X_test:
    distances = distance_ecu(X_train, X_test_point)   # step 1
    nearest_idx = nearest_neighbor(distances, K)      # step 2
    y_pred_point = voting(nearest_idx, y_train)       # step 3
    y_pred.append(y_pred_point)

print("Predictions:", y_pred)




from collections import Counter
import numpy as np

def KNN_from_scratch(X_train, y_train, X_test, K):
    y_pred = []
    for test_point in X_test:
        # Compute Euclidean distances from the test point to all training points
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))

        # Find the indices of the K nearest neighbors
        nearest_neighbor_ids = distances.argsort()[:K]

        # Get the labels of the nearest neighbors
        nearest_labels = y_train[nearest_neighbor_ids]

        # Majority vote
        most_common = Counter(nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])

    return np.array(y_pred)

# Now you can call it
K = 3
y_pred_scratch = KNN_from_scratch(normalized_X_train, y_train, normalized_X_test, K)
print(y_pred_scratch)








