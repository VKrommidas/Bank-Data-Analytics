# Bank Data Analytics

A Python-based data analytics project using the Bank Marketing dataset. The project explores customer data, applies clustering algorithms for customer segmentation, and develops machine learning models to predict whether a client subscribes to a term deposit.

## Project Overview

The project is organized into three stages:

### 1. Exploratory Data Analysis

- Load and inspect the Bank Marketing dataset.
- Examine missing values and unknown entries.
- Calculate descriptive statistics for numerical variables.
- Visualize data distributions using histograms and boxplots.
- Standardize numerical features.

### 2. Customer Segmentation - Clustering

- Select relevant numerical features: age, balance, and duration.
- Apply feature scaling using StandardScaler.
- Perform customer segmentation using:
  - K-Means clustering
  - DBSCAN clustering
- Evaluate clustering performance using the Silhouette Score.
- Visualize customer clusters.

### 3. Predictive Modeling - Classification

Develop and evaluate classification models to predict term deposit subscription.

The implemented models are:

- Gaussian Naive Bayes
- Calibrated Linear Support Vector Machine (SVM)
- Artificial Neural Network using TensorFlow/Keras
- Neural Network initialized with weights transferred from the first network

Model performance is evaluated using:

- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC-AUC Score
- ROC Curves

## Model Performance

The following results were obtained during one execution of the classification pipeline:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Gaussian Naive Bayes | 0.86 | 0.8151 |
| Calibrated Linear SVM | 0.90 | 0.9071 |
| Neural Network | 0.91 | 0.9270 |
| Neural Network with Transferred Weights | 0.91 | 0.9282 |

Results may vary between executions, particularly for the neural network models.

## Dataset

The project uses the Bank Marketing dataset (`bank-full.csv`), containing information about direct marketing campaigns conducted by a Portuguese banking institution.

The dataset includes customer characteristics, campaign-related information, and the target variable `y`, indicating whether a client subscribed to a term deposit.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Project Structure

```text
Bank-Data-Analytics/
|
|-- Step 1.py
|-- Step 2.py
|-- Step 3.py
|-- bank-full.csv
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Bank-Data-Analytics
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the scripts

Execute the stages individually:

```bash
python "Step 1.py"
python "Step 2.py"
python "Step 3.py"
```

## Important Note

The classification stage includes `duration` as a predictive feature. Since call duration is only known after a marketing call has ended, the resulting classification performance should not be interpreted as a realistic prediction made before the call.

## Author

Vasileios Krommydas