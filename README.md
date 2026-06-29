# Machine Learning

Welcome to the **Machine Learning** repository! This repository showcases various assignments I've worked on, implementing fundamental and advanced machine learning models.

- **1D Linear Regression**
- **N-Dimensional Linear Regression**
- **Polynomial Regression**
- **Classification & Clustering - Spaceship Data**
- **RAG & Agents**

## 👁️ Computer Vision & Object Detection (YOLO)

**Overview:** This project focuses on building an object detection pipeline using Python and YOLO (You Only Look Once). The Jupyter Notebook demonstrates the end-to-end process of preparing image data, running object detection models, and evaluating the results using modern computer vision techniques.

**Key Technologies:**
* **Language:** Python
* **Libraries:** YOLO (Ultralytics), OpenCV, Matplotlib, Pandas
* **Concepts:** Object Detection, Image Processing, Bounding Box Visualization, Model Evaluation

**Features:**
* Implemented a robust object detection model to identify and classify specific targets within image datasets.
* Processed and visualized image data using OpenCV and Matplotlib.
* Evaluated model accuracy and performance metrics directly within a Jupyter Notebook environment.

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

## RAG & Agents

This project is split into two notebooks exploring **Retrieval-Augmented Generation (RAG)** and **LLM Agents** using state-of-the-art transformer models and the Google Agent Development Kit (ADK).

### RAG (Retrieval-Augmented Generation)

Implements a full RAG pipeline from scratch — embedding a corpus with a transformer model, retrieving relevant passages via cosine similarity, and generating grounded answers with an LLM.

#### Datasets:
- **Winnie The Pooh** (literary text): Paragraphs extracted and cleaned from the full-text book.
- **Diseases & Symptoms** (Hugging Face): Structured medical records describing disease names, symptoms, and treatments.

#### Embedding Models:
- **`google/embeddinggemma-300m`** via `SentenceTransformer` — a 300M-parameter transformer used to encode both documents and queries into dense vector representations.
- **Voyage AI** (`voyageai`) — cloud embedding API used as a second embedder for comparison against the local HuggingFace model.

#### Key Steps:
- **Data Preparation**: Text splitting into paragraphs, regex-based cleaning, and EDA (word counts, character frequencies, paragraph length distributions).
- **Encoding**: Documents are encoded using `encode_document` and queries with `encode_query`, both normalized for cosine similarity search.
- **Vector Search**: Cosine similarity between the query vector and all document embeddings; top-K most relevant paragraphs are retrieved.
- **Generation**: Retrieved context is injected into a prompt and passed to:
  - **TinyLlama 1.1B** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) — local HuggingFace pipeline with streaming output.
  - **Gemini API** (`gemini-3.5-flash`, `gemini-2.5-flash`) — cloud LLM for higher-quality generation.
- **Experiments**: Varied K (1, 5, 10 retrieved paragraphs) and temperature (1.0 vs 0.5), tested on both related and unrelated queries to evaluate retrieval relevance and hallucination resistance.

#### Technologies & Libraries:
- `sentence-transformers`, `transformers`: Embedding and generation models
- `voyageai`: Cloud embedding API
- `google-genai`: Gemini API client
- `nltk`, `pandas`, `matplotlib`, `seaborn`: EDA and visualization

---

### Agents (Google ADK Stock Analysis Agent)

Builds a **Stock Analysis Agent** using the Google Agent Development Kit (ADK) powered by Gemini via LiteLLM. The agent orchestrates a multi-tool pipeline to analyze, compare, and visualize stock performance on demand.

#### Tools:
1. **Company Identifier** — extracts company names from free-text user queries.
2. **Ticker Resolver** — resolves company names to stock ticker symbols via Yahoo Finance's autocomplete API.
3. **Historical Stock Data** — downloads OHLC price history using `yfinance` and stores it in an in-memory cache.
4. **Calculate Metrics** — computes min/max/average price and percentage return from cached data without exposing raw data to the LLM.
5. **Visualize Stocks** — generates configurable multi-panel charts: raw price + 20-day SMA, normalized growth (base 100), and daily volatility (% returns).

#### Agent Design:
- **Skill file** (`SKILL.md`): Provides the agent with a structured playbook defining the exact tool-call sequence for analysis, visualization, and error-handling scenarios.
- **Safety settings**: Blocks dangerous content; temperature set to 0.2 for consistent, deterministic responses.
- **Guardrails**: Agent refuses non-financial queries and handles edge cases such as unknown companies, missing tickers, and unavailable historical data.

#### Tests:
- Multi-stock comparison and return ranking (Apple vs. Microsoft, Tesla vs. Nvidia)
- Incremental follow-up queries (normalized charts for Amazon, Nvidia, Microsoft)
- Error handling for fictional tickers ("Peaky Pookie") and low-liquidity stocks (Osem)
- Guardrails against irrelevant and harmful requests

#### Technologies & Libraries:
- `google-adk`: Agent Development Kit (LlmAgent, Runner, FunctionTool, InMemorySessionService)
- `litellm`: Unified LLM gateway for Gemini
- `yfinance`: Historical stock data
- `requests`: Yahoo Finance ticker resolution
- `pandas`, `matplotlib`: Data processing and visualization

---

## Installation

To get started with the models in this repository, you'll need to have Python installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn keras xgboost
```

## About

This repository is a comprehensive collection of machine learning assignments showcasing various techniques and algorithms applied to real-world datasets.
