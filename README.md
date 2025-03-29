# Machine Learning

Welcome to the **Machine Learning** repository! This repository showcases various assignments I've worked on, implementing fundamental machine learning models. The following models are included:

- **1D Linear Regression**
- **N-Dimensional Linear Regression**
- **Polynomial Regression**

Each model is implemented in Python, utilizing libraries like `numpy`, `pandas`, `matplotlib`, and `scikit-learn` for data manipulation, visualization, and model building. Below is an overview of each notebook:

## 1D Linear Regression

This notebook demonstrates the implementation of **Linear Regression** for a single feature. Key concepts covered include:

- **Data Preprocessing**: Importing and cleaning the dataset.
- **Model Creation**: Building the linear regression model.
- **Gradient Descent Optimization**: Training the model using gradient descent.
- **Prediction**: Making predictions on unseen data.

Key Steps:
- Loading and visualizing data.
- Fitting the linear regression model using the least squares method.
- Training the model using gradient descent.
- Visualizing the linear fit.

## N-Dimensional Linear Regression

In this notebook, I extend the **Linear Regression** model to work with datasets that have multiple features (N-Dimensions). This model is capable of handling higher-dimensional data and provides insights into the relationships between multiple features and the target variable.

Key Steps:
- Data preprocessing (scaling and handling missing values).
- Training the model with multiple features.
- Evaluating performance using metrics like Mean Squared Error (MSE).
- Visualizing the predicted vs actual values in N-dimensional space.

## Polynomial Regression Model Selection

This notebook demonstrates **Polynomial Regression** and how to select the best polynomial degree for fitting the data. I use **Mean Squared Error (MSE)** to evaluate the model's performance and choose the optimal degree.

Key Steps:
- Data visualization and exploration.
- Implementing polynomial regression with different degrees.
- Comparing the performance of various models using MSE.
- Visualizing the polynomial fits to determine the best degree.

## Encrypted Message Decoding

This section extends Part A by **learning a frequency-to-character mapping** from labeled header signals, then decoding messages using this custom mapping.

### Process:
1. **Training Phase**:
    - Analyze header signals to find optimal frequencies (1–48) via MSE minimization.
    - Save learned frequency-symbol pairs to `Learned_Symbols2Freqs.csv`.
2. **Decoding Phase**:
    - Use the learned CSV to decode unknown messages, repeating MSE-based frequency selection.

### Key Features:
- **Adaptability**: Infers mappings without predefined rules.
- **Consistency**: Maintains MSE evaluation and visualization (signal/decision plots) from Part A.
- **Output**: Decoded message (e.g., `MACHINE_LEARNING`) with frequencies and symbols from the learned mapping.

This end-to-end pipeline highlights model generalization from training data to real-world decoding tasks.

## Installation

To get started with the models in this repository, you'll need to have Python installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```
