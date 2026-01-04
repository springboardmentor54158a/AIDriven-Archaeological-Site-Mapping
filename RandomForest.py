import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
df=pd.read_csv("/content/Terrain.csv")
df.head()
df.columns
X = df.drop(columns=["Label", "Soil_Erosion_Rate"])
y_class = df["Label"]
y_reg = df["Soil_Erosion_Rate"]
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_class_test, y_class_pred))
print("\nClassification Report:\n")
print(classification_report(y_class_test, y_class_pred))
cm = confusion_matrix(y_class_test, y_class_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Erosion Classification Confusion Matrix")
plt.show()
reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_train, y_reg_train)
y_reg_pred = reg.predict(X_test)

rmse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("RMSE:", rmse)
print("R² Score:", r2)
plt.figure(figsize=(6,5))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.6)
plt.xlabel("Actual Soil Erosion Rate")
plt.ylabel("Predicted Soil Erosion Rate")
plt.title("Erosion Rate Prediction")
plt.show()
feature_importance = pd.Series(
    reg.feature_importances_,
    index=df.drop(columns=["Label", "Soil_Erosion_Rate"]).columns
).sort_values(ascending=False)

plt.figure(figsize=(8,6))
feature_importance.head(10).plot(kind="barh")
plt.title("Top Factors Affecting Erosion")
plt.gca().invert_yaxis()
plt.show()
