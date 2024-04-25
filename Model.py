import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv('Depression.csv')

# Split data into features (x) and target variable (y)
x = data.drop('Severtiy Level', axis=1)
y = data["Severtiy Level"]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Creating pipeline with StandardScaler and DecisionTreeClassifier
pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier())
pipeline.fit(x_train, y_train)

# Cross-validation
cv_scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring="accuracy")
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

# Model evaluation
y_pred_train = pipeline.predict(x_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training set accuracy:", train_accuracy)

y_pred_test = pipeline.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test set accuracy:", test_accuracy)

# Overall model accuracy
overall_accuracy = accuracy_score(y, model.predict(x))
print("Overall model accuracy:", overall_accuracy)


with open('Checkmodel.pkl', 'wb') as file:
  pickle.dump(model,file)
