import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Loading the data
file_path = 'bank-full.csv'
data = pd.read_csv(file_path, sep=';')
# Inspect the original dataset before cleaning
print("Original dataset shape:", data.shape)

print("\nMissing values:")
print(data.isna().sum())

print("\nUnknown values per column:")
print((data == 'unknown').sum())

print("\nRows containing at least one unknown value:",
      (data == 'unknown').any(axis=1).sum())

# Replacing 'unknown' values ​​with NaN and removing these rows
#data.replace('unknown', np.nan, inplace=True)
#data.dropna(inplace=True)
data = data.dropna()

# Dataset Overview
print("Dataset shape:", data.shape)\

print("\nMissing values:")
print(data.isnull().sum())

# Simple statistical analysis in the form of histograms and box plots for the key characteristics
numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Descriptive statistics
print("\nDescriptive statistics:")
print(data[numerical_features].describe())


# Histograms for the numerical features
plt.figure(figsize=(20, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# Box plots for the numerical features
plt.figure(figsize=(20, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=data[feature])
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# Normalization of numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

#Print the first five rows of the new dataset
print(data.head())
