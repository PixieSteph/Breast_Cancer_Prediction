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
