import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle as p1

# Load optdigits dataset from GitHub
optdigits = pd.read_csv("../lab 02/datasets/optdigits.tes", sep=",", header=None)

# Extract features and target
X = optdigits.iloc[:, :-1]  # Features are all columns except the last one
y = optdigits.iloc[:, -1]   # Target is the last column

# Load training data from file
XX = p1.load(open('../lab 02/esp', 'rb'))
print(type(XX))
print(XX.shape)


# Use the same data for testing
X_test, y_test = X, y

# Define the KNN classifier
n_neighbors = 15
clf = KNeighborsClassifier(n_neighbors)

# Train the KNN classifier with the training data
clf.fit(XX, y)

# Make predictions on the training and testing data
y_pred_train = clf.predict(XX)
y_pred_test = clf.predict(X_test)

# Calculate the accuracy of the classifier on the training and testing data
accuracy_train = accuracy_score(y, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

# Print a classification report
print(classification_report(y_test, y_pred_test))
