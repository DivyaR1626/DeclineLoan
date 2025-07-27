# -----------------------------
# Project 1: Decline Curve Forecasting
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample production data (time in months, production in STB/day)
time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
production = np.array([1000, 850, 720, 610, 520, 440, 380, 330, 290, 260, 230])

# Exponential Decline
def exp_decline(t, qi, Di):
    return qi * np.exp(-Di * t)

# Hyperbolic Decline
def hyp_decline(t, qi, Di, b):
    return qi / (1 + b * Di * t)**(1/b)

# Fit the model
popt_exp, _ = curve_fit(exp_decline, time, production, bounds=(0, [2000, 1]))
popt_hyp, _ = curve_fit(hyp_decline, time, production, bounds=(0, [2000, 1, 2]))

# Plot
plt.figure(figsize=(10,6))
plt.scatter(time, production, label='Observed', color='black')
plt.plot(time, exp_decline(time, *popt_exp), label='Exponential', linestyle='--')
plt.plot(time, hyp_decline(time, *popt_hyp), label='Hyperbolic', linestyle='-.')
plt.title('Decline Curve Forecasting')
plt.xlabel('Time (months)')
plt.ylabel('Production Rate (STB/day)')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Project 2: Loan Eligibility Prediction
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample DataFrame
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'ApplicantIncome': [5000, 3000, 4000, 2500, 6000],
    'LoanAmount': [150, 100, 130, 80, 160],
    'Loan_Status': ['Y', 'N', 'Y', 'N', 'Y']
})

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Features and Target
X = data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount']]
y = data['Loan_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
