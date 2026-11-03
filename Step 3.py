import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pathlib import Path

file_path = Path(__file__).parent / "bank-full.csv"
data = pd.read_csv(file_path, sep=";")


data = data.dropna()

# Conversion of the variable 'y' into numeric format
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Preparation of features and labels
X = data.drop('y', axis=1)
y = data['y']

# Conversion of categorical features into binary variables
X = pd.get_dummies(X, drop_first=True)

# Splitting data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Part 1: Scikit-Learn Classifiers
# -------------------------
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


# Naive Bayes Training
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)
y_pred_nb = nb_classifier.predict(X_test_scaled)
y_prob_nb = nb_classifier.predict_proba(X_test_scaled)[:, 1]

# Linear SVM Training with probability calibration
svm_classifier = CalibratedClassifierCV(
    estimator=LinearSVC(random_state=42, max_iter=10000),
    method='sigmoid',
    cv=3,
    ensemble=False
)

svm_classifier.fit(X_train_scaled, y_train)
y_pred_svm = svm_classifier.predict(X_test_scaled)
y_prob_svm = svm_classifier.predict_proba(X_test_scaled)[:, 1]

# Evaluation of the classifiers
print("Naive Bayes Classifier:")
print(classification_report(y_test, y_pred_nb))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nb)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_nb)}")

print("\nSVM Classifier:")
print(classification_report(y_test, y_pred_svm))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_svm)}")

# ROC Curve for the classifiers
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

plt.figure()
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC: {:.2f})'.format(roc_auc_score(y_test, y_prob_nb)))
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC: {:.2f})'.format(roc_auc_score(y_test, y_prob_svm)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# -------------------------
# Part 2: TensorFlow/Keras Neural Networks
# -------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Neural network (without transfer learning)
model = Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0,
                    callbacks=[EarlyStopping(patience=3)])

# Evaluation
y_prob_keras = model.predict(X_test_scaled, verbose=0).ravel()
y_pred_keras = (y_prob_keras > 0.5).astype("int32")

print("\nKeras Neural Network (without transfer learning):")
print(classification_report(y_test, y_pred_keras))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_keras)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_keras)}")

# Neural network (with transfer learning)
model_transfer = Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_transfer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Transfer of weights from the first model
for i in range(len(model.layers)-1):
    model_transfer.layers[i].set_weights(model.layers[i].get_weights())

history_transfer = model_transfer.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0,
                                      callbacks=[EarlyStopping(patience=3)])

# Evaluation
y_prob_transfer = model_transfer.predict(X_test_scaled, verbose=0).ravel()
y_pred_transfer = (y_prob_transfer > 0.5).astype("int32")

print("\nKeras Neural Network (with transfer learning):")
print(classification_report(y_test, y_pred_transfer))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_transfer)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_transfer)}")

# # ROC Curve for neural networks
fpr_keras, tpr_keras, _ = roc_curve(y_test, y_prob_keras)
fpr_transfer, tpr_transfer, _ = roc_curve(y_test, y_prob_transfer)

plt.figure()
plt.plot(fpr_keras, tpr_keras, label='Keras (AUC: {:.2f})'.format(roc_auc_score(y_test, y_prob_keras)))
plt.plot(fpr_transfer, tpr_transfer, label='Keras Transfer Learning (AUC: {:.2f})'.format(roc_auc_score(y_test, y_prob_transfer)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Keras Models')
plt.legend()
plt.show()
