import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your data
df = pd.read_csv("data.csv")

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Create new 'year', 'month', and 'day' columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop the 'Referee', 'location', and 'date' columns
df = df.drop(['Referee', 'location', 'date'], axis=1)

# Convert categorical data into numbers using one-hot encoding
df = pd.get_dummies(df, columns=['B_Stance', 'R_Stance', 'weight_class'])

# Decide which columns to use as features
features = df.drop(['R_fighter', 'B_fighter', 'Winner'], axis=1)
outcome = df['Winner']

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=5000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Assign the logistic regression model from the pipeline to logistic_model
logistic_model = pipeline['classifier']

# Use the model to make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

confusion_mat = confusion_matrix(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_mat)

# Perform cross-validation
# cv_scores = cross_val_score(pipeline, features, outcome, cv=5)
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Accuracy:", np.mean(cv_scores))


# Get user input for the names of the fighters
fighter_1 = input("Enter the name of Fighter 1: ")
fighter_2 = input("Enter the name of Fighter 2: ")

# Extract the rows corresponding to the new fighters from the dataset
new_fighter_1_data = df[df['R_fighter'] == fighter_1].drop(['R_fighter', 'B_fighter', 'Winner'], axis=1)
new_fighter_2_data = df[df['R_fighter'] == fighter_2].drop(['R_fighter', 'B_fighter', 'Winner'], axis=1)

# Check if the fighter is not in the 'R_fighter' column
if new_fighter_1_data.empty:
    new_fighter_1_data = df[df['B_fighter'] == fighter_1].drop(['R_fighter', 'B_fighter', 'Winner'], axis=1)

if new_fighter_2_data.empty:
    new_fighter_2_data = df[df['B_fighter'] == fighter_2].drop(['R_fighter', 'B_fighter', 'Winner'], axis=1)

# Check if the fighters are not in the dataset
if new_fighter_1_data.empty or new_fighter_2_data.empty:
    print("One or both fighters are not found in the dataset.")
else:
    # Concatenate the data for the two fighters into a single dataframe
    new_data = pd.concat([new_fighter_1_data, new_fighter_2_data], axis=0)

    # Preprocess the new data
    new_data = pd.get_dummies(new_data)

    # Realign the columns in new_data with the columns in features
    new_data = new_data.reindex(columns=features.columns, fill_value=0)

    # Transform the new data using the same imputer and scaler used during training
    new_data = pipeline['imputer'].transform(new_data)
    new_data = pipeline['scaler'].transform(new_data)

    # Use the trained model to get probability estimates for the new data
    new_probabilities = logistic_model.predict_proba(new_data)

    # Calculate the winning probabilities for each fighter
    fighter_1_win_prob = new_probabilities[0].max()
    fighter_2_win_prob = new_probabilities[1].max()

    # Normalize the winning probabilities
    total_win_prob = fighter_1_win_prob + fighter_2_win_prob
    fighter_1_win_percent = (fighter_1_win_prob / total_win_prob) * 100
    fighter_2_win_percent = (fighter_2_win_prob / total_win_prob) * 100

    # Round the winning percentages to two decimal places
    fighter_1_win_percent = round(fighter_1_win_percent, 2)
    fighter_2_win_percent = round(fighter_2_win_percent, 2)

    # Output the winning probabilities in the desired format
    print("Fighter 1 has a {:.2f}% chance to win".format(fighter_1_win_percent))
    print("Fighter 2 has a {:.2f}% chance to win".format(fighter_2_win_percent))

