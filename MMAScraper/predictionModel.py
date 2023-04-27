import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load data from CSV file
df = pd.read_csv('data.csv')

# Clean and preprocess data as needed
# For example, drop irrelevant columns
relevant_columns = [
    'R_fighter', 'B_fighter', 'Winner', 'B_avg_KD', 'B_avg_opp_KD', 'B_avg_SIG_STR_pct', 'B_avg_opp_SIG_STR_pct',
    'R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct', 'B_age', 'R_age', 'B_Height_cms',
    'R_Height_cms', 'B_Reach_cms', 'R_Reach_cms', 'B_total_rounds_fought', 'R_total_rounds_fought'
]
df = df[relevant_columns]

# Drop rows with missing values if applicable
df.dropna(inplace=True)

# Calculate feature differences
df['KD_diff'] = df['B_avg_KD'] - df['R_avg_KD']
df['opp_KD_diff'] = df['B_avg_opp_KD'] - df['R_avg_opp_KD']
df['SIG_STR_pct_diff'] = df['B_avg_SIG_STR_pct'] - df['R_avg_SIG_STR_pct']
df['opp_SIG_STR_pct_diff'] = df['B_avg_opp_SIG_STR_pct'] - df['R_avg_opp_SIG_STR_pct']
df['age_diff'] = df['B_age'] - df['R_age']
df['height_diff'] = df['B_Height_cms'] - df['R_Height_cms']
df['reach_diff'] = df['B_Reach_cms'] - df['R_Reach_cms']
df['total_rounds_fought_diff'] = df['B_total_rounds_fought'] - df['R_total_rounds_fought']

# Split data into features (X) and target (y) variables
X = df[['KD_diff', 'opp_KD_diff', 'SIG_STR_pct_diff', 'opp_SIG_STR_pct_diff',
         'age_diff', 'height_diff', 'reach_diff', 'total_rounds_fought_diff']]
y = df['Winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Instantiate the GridSearchCV with the model and hyperparameter grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid_search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


def get_prediction():
    blue_fighter = blue_fighter_entry.get()
    red_fighter = red_fighter_entry.get()

    # Perform data preprocessing on user input
    blue_fighter_data = df[df['B_fighter'] == blue_fighter]
    red_fighter_data = df[df['R_fighter'] == red_fighter]

    if not blue_fighter_data.empty and not red_fighter_data.empty:
        # Calculate feature differences for user input
        KD_diff = blue_fighter_data['KD_diff'].values[0]
        opp_KD_diff = blue_fighter_data['opp_KD_diff'].values[0]
        SIG_STR_pct_diff = blue_fighter_data['SIG_STR_pct_diff'].values[0]
        opp_SIG_STR_pct_diff = blue_fighter_data['opp_SIG_STR_pct_diff'].values[0]
        age_diff = blue_fighter_data['B_age'].values[0] - red_fighter_data['R_age'].values[0]
        height_diff = blue_fighter_data['B_Height_cms'].values[0] - red_fighter_data['R_Height_cms'].values[0]
        reach_diff = blue_fighter_data['B_Reach_cms'].values[0] - red_fighter_data['R_Reach_cms'].values[0]
        total_rounds_fought_diff = blue_fighter_data['B_total_rounds_fought'].values[0] - red_fighter_data['R_total_rounds_fought'].values[0]

        # Create input array and standardize the features
        input_array = [[KD_diff, opp_KD_diff, SIG_STR_pct_diff, opp_SIG_STR_pct_diff,
                        age_diff, height_diff, reach_diff, total_rounds_fought_diff]]
        input_array = scaler.transform(input_array)

        # Make prediction
        prediction = best_model.predict(input_array)
        prediction_proba = best_model.predict_proba(input_array)

        # Get the probability of the predicted class
        prediction_index = list(best_model.classes_).index(prediction[0])
        prediction_probability = prediction_proba[0][prediction_index]

        # Update the prediction_label text
        prediction_label.config(text=f"Winner: {prediction[0]} (Probability: {prediction_probability:.2%})")

    else:
        prediction_label.config(text="Error: One or both fighters not found in the dataset.")

# Create tkinter window
root = tk.Tk()
root.title("Fight Prediction")

# Create labels
blue_fighter_label = tk.Label(root, text="Blue Fighter:")
blue_fighter_label.grid(row=0, column=0, padx=10, pady=10)

red_fighter_label = tk.Label(root, text="Red Fighter:")
red_fighter_label.grid(row=1, column=0, padx=10, pady=10)

prediction_label = tk.Label(root, text="")
prediction_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Create entry boxes
blue_fighter_entry = tk.Entry(root)
blue_fighter_entry.grid(row=0, column=1, padx=10, pady=10)

red_fighter_entry = tk.Entry(root)
red_fighter_entry.grid(row=1, column=1, padx=10, pady=10)

# Create button
predict_button = tk.Button(root, text="Predict", command=get_prediction)
predict_button.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()