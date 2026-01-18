import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# 1. LOAD DATASET
# ================================
data_path = "data/erosion_dataset_augmented.csv"
df = pd.read_csv(data_path)

# ================================
# 2. ENCODE CATEGORICAL FEATURES
# ================================
le_veg = LabelEncoder()
le_terrain = LabelEncoder()

df["Vegetation"] = le_veg.fit_transform(df["Vegetation"])
df["Terrain_Type"] = le_terrain.fit_transform(df["Terrain_Type"])

# ================================
# 3. DEFINE FEATURES & TARGET
# ================================
X = df[["Slope", "Elevation", "Vegetation", "Terrain_Type"]]
y = df["Erosion_Score"]   # Continuous value (Regression)

# ================================
# 4. TRAIN–TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ================================
# 5. TRAIN RANDOM FOREST REGRESSOR
# ================================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 6. PREDICTION
# ================================
y_pred = model.predict(X_test)

# ================================
# 7. EVALUATION
# ================================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION (REGRESSION) =====")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")
