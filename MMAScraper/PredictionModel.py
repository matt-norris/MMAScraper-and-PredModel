import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data from CSV file
df = pd.read_csv('data.csv')

# Add new features
df['win_ratio_diff'] = (df['B_wins'] / (df['B_wins'] + df['B_losses'])) - (
            df['R_wins'] / (df['R_wins'] + df['R_losses']))
df['form_diff'] = df['B_current_win_streak'] - df['R_current_win_streak']

# Update relevant_columns list with new features
relevant_columns = [
    'R_fighter', 'B_fighter', 'Winner', 'B_avg_KD', 'B_avg_opp_KD', 'B_avg_SIG_STR_pct', 'B_avg_opp_SIG_STR_pct',
    'R_avg_KD', 'R_avg_opp_KD', 'R_avg_SIG_STR_pct', 'R_avg_opp_SIG_STR_pct', 'B_age', 'R_age', 'B_Height_cms',
    'R_Height_cms', 'B_Reach_cms', 'R_Reach_cms', 'B_total_rounds_fought', 'R_total_rounds_fought',
    'win_ratio_diff', 'form_diff'
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

# Define feature columns
feature_columns = [
    'KD_diff', 'opp_KD_diff', 'SIG_STR_pct_diff', 'opp_SIG_STR_pct_diff',
    'age_diff', 'height_diff', 'reach_diff', 'total_rounds_fought_diff',
]

# Split data into features (X) and target (y) variables
# Convert string class labels to numeric values
le = LabelEncoder()
df['Winner'] = le.fit_transform(df['Winner'])

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

# Perform feature selection using Recursive Feature Elimination (RFE)
model = XGBClassifier(random_state=42)
rfe = RFE(estimator=model, n_features_to_select=8)
rfe.fit(X_train, y_train)

# Transform X_train and X_test based on the selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Train an XGBoost model with hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(100, 1001, 100),
    'learning_rate': np.logspace(-3, 0, 10),
    'max_depth': np.arange(1, 11),
    'min_child_weight': np.arange(1, 11),
    'gamma': np.linspace(0, 1, 11),
    'subsample': np.linspace(0.5, 1, 6),
    'colsample_bytree': np.linspace(0.5, 1, 6)
}

model = XGBClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_dist, n_iter=100, cv=5,
    scoring='accuracy', n_jobs=-1, verbose=2, random_state=42
)

random_search.fit(X_train_rfe, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_rfe)
accuracy = accuracy_score(y_test, y_pred)

# Get the names of the selected features
selected_features = np.array(feature_columns)[rfe.support_]


def get_prediction():
    blue_fighter = blue_fighter_entry.get()
    red_fighter = red_fighter_entry.get()

    # Perform data preprocessing on user input
    blue_fighter_data = df[df['B_fighter'] == blue_fighter]
    red_fighter_data = df[df['R_fighter'] == red_fighter]

    if not blue_fighter_data.empty and not red_fighter_data.empty:
        # Calculate feature differences for user input
        input_data = blue_fighter_data[feature_columns].iloc[0] - red_fighter_data[feature_columns].iloc[0]

        # Create input array and standardize the features
        input_array = input_data[selected_features].values.reshape(1, -1)
        input_array = scaler.transform(input_array)

        # Make prediction
        prediction = best_model.predict(input_array)
        prediction_proba = best_model.predict_proba(input_array)

        # Get the probability of the predicted class
        prediction_index = list(best_model.classes_).index(prediction[0])
        prediction_probability = prediction_proba[0][prediction_index]

        # Update the prediction_label text
        prediction_label.config(
            text=f"Winner: {le.inverse_transform(prediction)[0]} (Probability: {prediction_probability:.2%})")

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

blue_fighter_entry = tk.Entry(root)
blue_fighter_entry.grid(row=0, column=1, padx=10, pady=10)

red_fighter_entry = tk.Entry(root)
red_fighter_entry.grid(row=1, column=1, padx=10, pady=10)

predict_button = tk.Button(root, text="Predict", command=get_prediction)
predict_button.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()
