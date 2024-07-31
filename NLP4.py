import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, mean_squared_error

# Download and extract SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Extract texts and labels
texts = df['message'].values
labels = df['label'].values

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)
y = labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Neural Network Classifier
nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)

# Evaluation Metrics
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)

accuracy_nn = accuracy_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)

# Print Evaluation Metrics
print("KNN Classifier:")
print(f"Accuracy: {accuracy_knn:.2f}")
print(f"F1 Score: {f1_knn:.2f}")
print(f"Precision: {precision_knn:.2f}")
print(f"MSE: {mse_knn:.2f}\n")

print("Neural Network Classifier:")
print(f"Accuracy: {accuracy_nn:.2f}")
print(f"F1 Score: {f1_nn:.2f}")
print(f"Precision: {precision_nn:.2f}")
print(f"MSE: {mse_nn:.2f}\n")

# Comparison of accuracies
methods = ['KNN', 'Neural Network']
accuracies = [accuracy_knn, accuracy_nn]
mses = [mse_knn, mse_nn]

# Plotting the bar chart for accuracy
plt.figure(figsize=(8, 5))
plt.bar(methods, accuracies, color=['blue', 'green'])
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of KNN and Neural Network')
plt.ylim(0, 1)
for i in range(len(accuracies)):
    plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center')
plt.show()

# Plotting the bar chart for MSE
plt.figure(figsize=(8, 5))
plt.bar(methods, mses, color=['blue', 'green'])
plt.xlabel('Method')
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison of KNN and Neural Network')
for i in range(len(mses)):
    plt.text(i, mses[i] + 0.01, f'{mses[i]:.2f}', ha='center')
plt.show()

# Sample messages for prediction
sample_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
    "Hey, are we still meeting for dinner tonight?",
    "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 12345.",
    "Call me when you get a chance, I need to discuss the project with you."
]

# Convert sample messages to TF-IDF features
sample_features = vectorizer.transform(sample_messages)

# Predict using KNN and Neural Network
knn_predictions = knn.predict(sample_features)
nn_predictions = nn.predict(sample_features)

# Print predictions
print("\nSample Message Predictions:")
for i, message in enumerate(sample_messages):
    print(f"Message: {message}")
    print(f"KNN Prediction: {'Spam' if knn_predictions[i] == 1 else 'True'}")
    print(f"Neural Network Prediction: {'Spam' if nn_predictions[i] == 1 else 'True'}")
    print()
