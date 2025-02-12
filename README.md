# Breast Cancer Prediction App
# Project Overview
This project leverages machine learning to predict whether a tumor is benign or malignant based on various input features related to breast cancer data. The application uses an Artificial Neural Network (ANN) model to make predictions and is deployed using Streamlit for interactive web usage.

The project also involves preprocessing the data, feature selection, and hyperparameter tuning to improve model performance. This app allows users to input tumor-related features and receive predictions on whether the tumor is benign or malignant.

# Technologies Used
1.Python: Programming language for building the app and machine learning model.
2.Streamlit: Framework for building the interactive web app.
3.scikit-learn: Machine learning library used for training models and preprocessing data.
4.pandas: For data manipulation and analysis.
5.numpy: For numerical operations.
6.matplotlib: For visualizing data and model performance.
7.sklearn.neural_network.MLPClassifier: For building the Artificial Neural Network model.

# Model Information
Model Type: Artificial Neural Network (ANN) using MLPClassifier.
Feature Selection:
Initially, all features were considered, and a SelectKBest technique was used for feature selection to choose the most relevant features.
The number of features (k) was selected based on model performance.

# Hyperparameters:
Optimized using GridSearchCV for parameters such as number of hidden layers, activation function, and solver.
Hyperparameter Tuning and Feature Selection
In this project, we explored different feature selection techniques and optimized the MLPClassifier using GridSearchCV. Some key configurations tested include:

# Hidden Layer Sizes: Configured with different numbers of neurons to check for overfitting and underfitting.
Activation Function: Explored options like 'relu' and 'tanh' to find the best performance.
Solver: Tried solvers like 'adam' and 'sgd' for better convergence.

# Important Notes
Data Scaling: All features are scaled using StandardScaler to improve model training, as neural networks work better when features are normalized.
Model Evaluation: After training the model, cross-validation was used to ensure the model generalizes well to unseen data.
Model Accuracy: The final model achieves an accuracy score of (insert accuracy here) on the test data.

# Future Improvements & Exploration
Model Exploration: You may explore alternative algorithms like Support Vector Machines (SVM) or Random Forests and compare them to the ANN model.
Feature Engineering: Consider adding more advanced techniques for feature engineering and exploring PCA for dimensionality reduction.
Model Deployment: Streamlit app could be hosted on platforms like Heroku or Streamlit Cloud for wider accessibility.



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
import pickle

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Feature Selection (Select Most Relevant Features)
k_best = SelectKBest(score_func=f_classif, k=10)
X_new = k_best.fit_transform(X, y)
selected_features = X.columns[k_best.get_support()]

# Fit StandardScaler on the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)  # StandardScaler is trained on the 10 features

#  Train the ANN model
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_scaled, y) 

# Save the trained model
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump((scaler, k_best, mlp, selected_features), f)

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is benign or malignant using an ANN model.")

# Sidebar for user input
st.sidebar.header("Enter Feature Values")

user_input = []
for feature in selected_features:
    value = st.sidebar.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    user_input.append(value)

# Predict button
if st.sidebar.button("Predict"):
    #  Load the model
    with open("breast_cancer_model.pkl", "rb") as f:
        scaler, k_best, mlp, selected_features = pickle.load(f)

    #  Convert input to NumPy array with correct shape
    user_data = np.array(user_input).reshape(1, -1)

    # Scale input 
    user_data_scaled = scaler.transform(user_data)

    # Make prediction using the trained MLP model
    prediction = mlp.predict(user_data_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"

    # Display prediction
    st.write(f"### **Prediction: {result}**")
