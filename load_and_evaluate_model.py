# load_and_evaluate_model.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_evaluate_model():
    # Load the trained model
    clf = joblib.load('breast_cancer_svm_model.pkl')

    # Load the breast cancer dataset
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Breast Cancer Classification')
    plt.show()

if __name__ == "__main__":
    load_and_evaluate_model()
