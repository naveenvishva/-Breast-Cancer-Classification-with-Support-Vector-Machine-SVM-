## Breast Cancer Classification with Support Vector Machine (SVM)

This repository contains Python scripts for training an SVM model on the Breast Cancer dataset and evaluating its performance.

### Files

1. **train_and_save_model.py**: Script for training an SVM model using the Breast Cancer dataset and saving the trained model.
2. **load_and_evaluate_model.py**: Script for loading the saved SVM model, evaluating its accuracy, and visualizing the results using a confusion matrix.
3. **requirements.txt**: File containing the dependencies required to run the scripts.
4. **breast_cancer_evaluate_model.py**: Additional evaluation script for the Breast Cancer classification model.

### How to Use

1. **Training the Model**:
   - Run `train_and_save_model.py` to train the SVM model on the Breast Cancer dataset and save the trained model as `breast_cancer_svm_model.pkl`.
   ```bash
   python train_and_save_model.py
   ```

2. **Evaluating the Model**:
   - Run `load_and_evaluate_model.py` to load the saved model, evaluate its accuracy on the test set, and visualize the results using a confusion matrix.
   ```bash
   python load_and_evaluate_model.py
   ```

### Dependencies

Ensure you have the required dependencies installed by running:
```bash
pip install -r requirements.txt
```

### Scripts Overview

- **train_and_save_model.py**:
  - Loads the Breast Cancer dataset.
  - Splits the dataset into training and testing sets.
  - Creates an SVM classifier with a linear kernel.
  - Trains the model and saves it as `breast_cancer_svm_model.pkl`.
  - Prints the accuracy of the trained model on the test set.
  
- **load_and_evaluate_model.py**:
  - Loads the saved SVM model.
  - Loads the Breast Cancer dataset.
  - Splits the dataset into training and testing sets.
  - Evaluates the accuracy of the loaded model on the test set.
  - Generates a confusion matrix and visualizes it.
  
### Additional Evaluation Script

You can also use the `breast_cancer_evaluate_model.py` script to evaluate the model separately.

### Requirements

- **joblib** (version 1.1.0)
- **numpy** (version 1.21.3)
- **scikit-learn** (version 0.24.2)

### License

This project is licensed under the [MIT License](LICENSE).
