# titanic-ML-
Feature Scaling and Normalization – a standardization for machine learning algorithms

What is Feature Scaling?
•Feature Scaling is a method to scale numeric features in the same scale or range (like:-1 to 1,  0 to 1).

•This is the last step involved in Data Preprocessing and before ML model training.

•It is also called as data normalization.

•We apply Feature Scaling on independent variables.

•We fit feature scaling with train data and transform on train and test data.

Why Feature Scaling?
•The scale of raw features is different according to its units.

•Machine Learning algorithms can’t understand features units, understand only numbers.

•Ex: If hight 140cm and 8.2feet

•ML Algorithms understand numbers then 140 > 8.2
Which ML Algorithms Required Feature Scaling?
Those Algorithms Calculate Distance
•K-Nearest Neighbors (KNN)

•K-Means

•Support Vector Machine (SVM)

•Principal Component Analysis(PCA)

•Linear Discriminant Analysis

Gradient Descent Based Algorithms
•Linear Regression,

•Logistic Regression

•Neural Network

Tree Based Algorithms not required Feature scaling
•Decision Tree, Random Forest, XGBoost
