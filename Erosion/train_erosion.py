import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/erosion_dataset_augmented.csv")

# Encode categorical features
le_veg = LabelEncoder()
le_terrain = LabelEncoder()

data["Vegetation"] = le_veg.fit_transform(data["Vegetation"])
data["Terrain_Type"] = le_terrain.fit_transform(data["Terrain_Type"])

# Select features and label
X = data[["Slope", "Elevation", "Vegetation", "Terrain_Type"]]
y = data["Erosion_Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("===== MODEL EVALUATION (CLASSIFICATION) =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
