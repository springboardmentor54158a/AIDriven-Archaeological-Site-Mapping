import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib

data = pd.read_csv("terrain_features.csv")

X = data[['slope', 'veg_index', 'elevation']]

y = data['erosion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

y_prob = model.predict_proba(X_test)[:, 1]

rmse = np.sqrt(mean_squared_error(y_test, y_prob))
r2 = r2_score(y_test, y_prob)

print("Model Evaluation Results")
print("------------------------")
print("Accuracy :", round(accuracy, 4))
print("RMSE     :", round(rmse, 4))
print("R2 Score :", round(r2, 4))

# Save trained model
joblib.dump(model, "erosion_rf_model.pkl")
print("\nRandom Forest model saved as erosion_rf_model.pkl")
