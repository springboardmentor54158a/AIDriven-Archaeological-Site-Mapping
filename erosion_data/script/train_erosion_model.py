import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib

# Load terrain feature dataset
data = pd.read_csv("terrain_features.csv")

# Input features
X = data[['slope', 'veg_index', 'elevation']]

# Target label (0 = Stable, 1 = Erosion-prone)
y = data['erosion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predict class labels (for accuracy)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Predict probabilities (for RMSE & R2)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute RMSE and R2 Score
rmse = np.sqrt(mean_squared_error(y_test, y_prob))
r2 = r2_score(y_test, y_prob)

# Print evaluation results
print("Model Evaluation Results")
print("------------------------")
print("Accuracy :", round(accuracy, 4))
print("RMSE     :", round(rmse, 4))
print("R2 Score :", round(r2, 4))

# Save trained model
joblib.dump(model, "erosion_rf_model.pkl")
print("\nRandom Forest model saved as erosion_rf_model.pkl")
