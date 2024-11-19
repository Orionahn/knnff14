![Me!](https://i.postimg.cc/qB55Y5QX/image-2024-11-19-063537440.png)

# FF14 K-NN Classification

This project implements a k-Nearest Neighbors (k-NN) classifier to predict player classifications (e.g., `Veteran Player` or `Non-Veteran Player`) based on player attributes. The model selects the best `k` value and scaler type (`MinMaxScaler` or `StandardScaler`) for optimal performance. The final model is evaluated using accuracy metrics and a confusion matrix, and it predicts the class of a new instance.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Required Python libraries:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

---

## Files in the Project

1. **`FF14 DATASET.xlsx`**: The dataset file used for training and testing the k-NN classifier.
   - Contains player attributes and their classifications (`Veteran Player` or `Non-Veteran Player`).
2. **`knn_classifier.py`**: The Python script that implements the k-NN model, performs preprocessing, evaluates performance, and predicts the class of a new instance.
3. **This README.md**: Instructions on how to run the project.

---

## Setup Instructions

### Step 1: Clone or Download the Project
1. Download the files to a folder on your computer.
2. Ensure the dataset file (`FF14 DATASET.xlsx`) is in the same directory as the Python script (`knn_classifier.py`).

### Step 2: Install Dependencies
Run the following command in your terminal to install the required libraries:
```bash
pip install pandas scikit-learn matplotlib
```
## Running the Code

### Step 1: Execute the Script
1. Open your terminal or command prompt.
2. Navigate to the directory containing the files.
3. Run the script using:
   ```bash
   python knn_classifier.py
   ```

### Step 2: Interpreting the Output
The script performs the following tasks:
1. **Data Preprocessing**:
   - Handles missing values in the dataset.
   - Scales data using either `MinMaxScaler` or `StandardScaler`.
2. **Model Training and Evaluation**:
   - Splits the dataset into training and test sets.
   - Tests multiple values of `k` (from 1 to 20) to find the best-performing configuration.
   - Outputs accuracy metrics and a confusion matrix.
3. **Prediction for a New Instance**:
   - Uses the trained model to classify a new player with the following attributes:
     - `Character Level`: 100
     - `Character Total Money`: 239232
     - `Alternate Characters Made`: 2
     - `Friends`: 200
     - `Rare Weapons Acquired`: 1
     - `Minions Collected`: 329
   - Prints the predicted class (encoded and original label).

---
## Notes

1. Ensure the dataset file (`FF14 DATASET.xlsx`) is properly formatted and matches the script's expectations:
- Columns should include `Instance`, `Veteran Player`, and all relevant feature attributes.
2. Modify the `new_instance` variable in the script if you want to classify another player.


