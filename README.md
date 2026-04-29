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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'cgpa': [6.8, 5.9, 5.3, 7.4, 5.8, 7.1, 6.5, 8.2, 5.0, 7.8],
    'iq': [123, 106, 121, 132, 142, 115, 98, 140, 110, 128],
    'placement': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
})

print("Dataset Preview:")
print(data.head())

X = data[['cgpa', 'iq']]
y = data['placement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = pd.DataFrame({
    'cgpa': [7.5],
    'iq': [120]
})

new_student = scaler.transform(new_student)
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nThe student is Placed")
else:
    print("\nThe student is Not Placed")


cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j])

plt.show()
 
*/
```

## Output:

<img width="1069" height="878" alt="Screenshot 2026-04-29 130357" src="https://github.com/user-attachments/assets/abd8504e-0bb1-4dc7-97b8-b209a085ab40" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
