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


