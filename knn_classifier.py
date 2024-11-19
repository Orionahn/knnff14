import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the dataset
file_path = 'FF14 DATASET.xlsx'  
dataset = pd.ExcelFile(file_path).parse(0)  # Load the first sheet

# Replace NaN values with 0 in the 'Rare Weapons Acquired' column (was having errors)
dataset['Rare Weapons Acquired'].fillna(0, inplace=True)

# Separate features and target
features = dataset.drop(columns=['Instance', 'Veteran Player'])
target = dataset['Veteran Player']

# Encode the target variable
encoder = LabelEncoder()
encoded_target = encoder.fit_transform(target)

# Task 1: Preprocessing, Training, and Evaluation with MinMaxScaler
print("Running Task 1 with MinMaxScaler...")
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_target, random_state=42) # get it..?

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training set
X_test_scaled = scaler.transform(X_test)       # Transform test set only

# Train the k-NN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy = knn.score(X_test_scaled, y_test)

print(f"Model Accuracy with MinMaxScaler: {accuracy * 100:.2f}%")

k_values = range(1, 21)

# Task 2(a): Analyze accuracy with varying k values (MinMaxScaler)
print("Running Task 2(a) with MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training set
X_test_scaled = scaler.transform(X_test)       # Transform test set only

minmax_train_accuracies = []
minmax_test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    minmax_train_accuracies.append(knn.score(X_train_scaled, y_train))  # Training accuracy
    minmax_test_accuracies.append(knn.score(X_test_scaled, y_test))    # Test accuracy

# Plot Task 2(a) results
plt.figure(figsize=(10, 6))
plt.plot(k_values, minmax_train_accuracies, label='Training Accuracy (MinMaxScaler)', marker='o')
plt.plot(k_values, minmax_test_accuracies, label='Test Accuracy (MinMaxScaler)', marker='o')
plt.title('k-NN Accuracy vs. k Value (MinMaxScaler)')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()

# Task 2(b): Analyze accuracy with varying k values (StandardScaler)
print("Running Task 2(b) with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training set
X_test_scaled = scaler.transform(X_test)       # Transform test set only

standard_train_accuracies = []
standard_test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    standard_train_accuracies.append(knn.score(X_train_scaled, y_train))
    standard_test_accuracies.append(knn.score(X_test_scaled, y_test))

# Plot Task 2(b) results
plt.figure(figsize=(10, 6))
plt.plot(k_values, standard_train_accuracies, label='Training Accuracy (StandardScaler)', marker='o')
plt.plot(k_values, standard_test_accuracies, label='Test Accuracy (StandardScaler)', marker='o')
plt.title('k-NN Accuracy vs. k Value (StandardScaler)')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()

# Task 3(a): Find the best value of k and best scaler
# Identify the best k and scaler based on test_accuracies
best_k_minmax = minmax_test_accuracies.index(max(minmax_test_accuracies)) + 1  # Best k for MinMaxScaler
best_k_standard = standard_test_accuracies.index(max(standard_test_accuracies)) + 1  # Best k for StandardScaler

# Select the best performing scaler
if max(minmax_test_accuracies) > max(standard_test_accuracies):
    best_k = best_k_minmax
    scaler = MinMaxScaler()
    print(f"Best k: {best_k} with MinMaxScaler")
else:
    best_k = best_k_standard
    scaler = StandardScaler()
    print(f"Best k: {best_k} with StandardScaler")

# Scale the data using the best scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training set
X_test_scaled = scaler.transform(X_test)       # Transform test set only

# Train the k-NN classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
accuracy = knn.score(X_test_scaled, y_test)
print(f"Final Model Accuracy with k={best_k}: {accuracy * 100:.2f}%")

# Task 3(b): Calculate the confusion matrix
y_pred = knn.predict(X_test_scaled)
confus_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(confus_matrix)

# Explanation of Confusion Matrix
"""
The confusion matrix is a 2x2 matrix that summarizes the model's performance:
- Rows represent instances in the actual class.
- Columns represent the instances in the predicted class.
- True predictions appear on the diagonal.
- Off-diagonal elements represent misclassifications.
"""

# Task 4: Apply your model to predict the class of a new instance

# Define the new instance with your values
new_instance = [[100, 239232, 2, 200, 1, 329]]  # Character Level, Money, Alternates, Friends, Rare Weapons, Minions
# Expected Value: Veteran Player

# Scale the new instance using the same scaler
scaled_instance = scaler.transform(new_instance)

# Predict the class of the new instance
predicted_class = knn.predict(scaled_instance)

# Decode the predicted class back to its original label
predicted_label = encoder.inverse_transform(predicted_class)

print("New Instance:", new_instance)
print("Predicted Class (Encoded):", predicted_class[0])
print("Predicted Class (Original Label):", predicted_label[0])

train_accuracy_final = knn.score(X_train_scaled, y_train)
test_accuracy_final = knn.score(X_test_scaled, y_test)
# Summarize results for final submission

summary_data = {
    "k": best_k,  
    "Scaler used for attribute preprocessing": "MinMaxScaler" if isinstance(scaler, MinMaxScaler) else "StandardScaler",
    "Model accuracy in the training set": train_accuracy_final,
    "Model accuracy in the test set": test_accuracy_final,
    "Confusion matrix": confus_matrix,
    "New instance attribute values": new_instance[0],  
    "Predicted class (Encoded)": int(predicted_class[0]),
    "Predicted class (Original Label)": predicted_label[0],
}
print("Summary of Results:")
for key, value in summary_data.items():
    print(f"{key}: {value}")
