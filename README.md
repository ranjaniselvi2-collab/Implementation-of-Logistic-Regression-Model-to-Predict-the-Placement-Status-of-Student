# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data** – Import dataset and separate features (X) and target (Placement).
2. **Split Data** – Divide data into training and testing sets.
3. **Train Model** – Fit Logistic Regression model using training data.
4. **Predict & Evaluate** – Predict placement on test data and calculate accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ranjani S
RegisterNumber: 212225230224

import numpy as np


X = np.array([[6.5, 1],
              [7.0, 2],
              [8.0, 2],
              [5.5, 0],
              [6.8, 1],
              [7.5, 2],
              [8.5, 3],
              [5.0, 0]])

y = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])

X = np.c_[np.ones(X.shape[0]), X]

weights = np.zeros((X.shape[1], 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


learning_rate = 0.01
epochs = 1000


for _ in range(epochs):
    z = np.dot(X, weights)             
    y_pred = sigmoid(z)                 
    
   
    gradient = np.dot(X.T, (y_pred - y)) / len(y)
   
    weights = weights - learning_rate * gradient

def predict(X_new):
    X_new = np.c_[np.ones(X_new.shape[0]), X_new]
    probs = sigmoid(np.dot(X_new, weights))
    return (probs >= 0.5).astype(int)


new_student = np.array([[7.2, 2]])
print("Prediction (1=Placed, 0=Not Placed):", predict(new_student)[0][0])

print("Final Weights:\n", weights)
 
*/
```

## Output:

<img width="1150" height="911" alt="Screenshot 2026-04-29 113302" src="https://github.com/user-attachments/assets/41f5df9d-734b-468f-96bc-76a1cf65bd90" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
