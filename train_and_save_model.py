# train_and_save_model.py

import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    # Load the breast cancer dataset
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear', C=2)
    clf.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(clf, 'breast_cancer_svm_model.pkl')

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

if __name__ == "__main__":
    train_and_save_model()
