# Machine Learning

Welcome to the **Machine Learning** repository! This repository showcases various assignments I've worked on, implementing fundamental and advanced machine learning models. The following models are included:

- **1D Linear Regression**
- **N-Dimensional Linear Regression**
- **Polynomial Regression**
- **Classification & Clustering - Spaceship Data**

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

## Classification & Clustering - Spaceship Data

This comprehensive final assignment demonstrates a complete machine learning pipeline applied to the Spaceship dataset from Kaggle. It showcases **Exploratory Data Analysis (EDA)**, **Feature Engineering**, and **Classification & Clustering** techniques.

### Dataset Overview:
The Spaceship dataset contains passenger information with the objective of predicting whether passengers were transported to an alternate dimension during a spaceship crisis. The dataset includes:
- **Passenger Demographics**: Age, Home Planet, Destination
- **Cabin Information**: Deck, Cabin Number, Side
- **Service Expenditure**: Expenses across RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- **Status Indicators**: CryoSleep, VIP status, Transportation status

### Key Sections:

#### 1. **Exploratory Data Analysis (EDA)**
- Statistical summaries and data distribution analysis
- Visualization of categorical and continuous features
- Identification of relationships between features and the target variable (Transported)
- Analysis of missing data patterns and their implications

#### 2. **Feature Engineering**
- **Expense Aggregation**: Combining individual expense categories into a total "Expenses" feature
- **Cabin Parsing**: Extracting deck, cabin number, and side from cabin addresses
- **Passenger Group Analysis**: Extracting group identifiers from passenger IDs
- **Categorical Encoding**: Converting categorical variables (Home Planet, Destination, Cabin Deck) to numerical representations

#### 3. **Data Preprocessing & Imputation**
- **High-Accuracy Imputation**: Using domain knowledge and logical rules to fill missing values
  - Expenses for CryoSleep passengers filled with 0
  - Expenses for children (Age < 13) filled with 0
  - HomePlanet inferred from Cabin Deck
  - CryoSleep status inferred from Cabin Deck
- **Statistical Imputation**: Using mean values grouped by categorical features
- **Interpolation**: Linear interpolation for numerical features like Cabin Number

#### 4. **Classification Models**
Multiple classification algorithms compared for predicting passenger transportation:
- k-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Gaussian Naive Bayes
- Decision Trees
- Random Forest
- Logistic Regression
- Support Vector Machines (SVM)
- XGBoost
- Neural Networks (Keras)

Model evaluation includes:
- Accuracy scores on train and test sets
- Confusion matrices and classification reports
- ROC curves and AUC metrics

#### 5. **Clustering Analysis**
- **K-Means Clustering**: Identifying natural groupings in passenger data
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering approach
- **Principal Component Analysis (PCA)**: Dimensionality reduction for visualization

### Key Steps:
- Comprehensive data exploration and visualization
- Intelligent feature engineering and domain-specific imputation
- Comparison of multiple machine learning algorithms
- Hyperparameter tuning for model optimization
- Evaluation using multiple metrics (accuracy, precision, recall, F1-score)
- Visualization of model performance through confusion matrices and ROC curves

### Technologies & Libraries:
- `pandas` & `numpy`: Data manipulation and numerical computing
- `matplotlib` & `seaborn`: Data visualization
- `scikit-learn`: Classification and clustering algorithms
- `keras`: Deep learning neural networks
- `xgboost`: Gradient boosting classifier

This project demonstrates a complete, production-ready machine learning workflow from raw data to model evaluation.

## Installation

To get started with the models in this repository, you'll need to have Python installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn keras xgboost
```

## About

This repository is a comprehensive collection of machine learning assignments showcasing various techniques and algorithms applied to real-world datasets.
