!pip install xgboost
from xgboost import XGBClassifier, XGBRegressor
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
xgb_clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_clf.fit(X_train, y_class_train)
y_pred_class = xgb_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_class_test, y_pred_class))
print(classification_report(y_class_test, y_pred_class))
cm = confusion_matrix(y_class_test, y_pred_class)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("XGBoost Erosion Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
xgb_reg = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_reg.fit(X_train, y_reg_train)
y_pred_reg = xgb_reg.predict(X_test)

rmse = mean_squared_error(y_reg_test, y_pred_reg)
r2 = r2_score(y_reg_test, y_pred_reg)

print("RMSE:", rmse)
print("R² Score:", r2)
plt.figure(figsize=(6,5))
plt.scatter(y_reg_test, y_pred_reg, alpha=0.6)
plt.xlabel("Actual Soil Erosion Rate")
plt.ylabel("Predicted Soil Erosion Rate")
plt.title("XGBoost Erosion Rate Prediction")
plt.show()
importance = xgb_reg.feature_importances_
features = df.drop(columns=["Label", "Soil_Erosion_Rate"]).columns

imp_df = pd.Series(importance, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,6))
imp_df.head(10).plot(kind="barh")
plt.title("Top Erosion Drivers (XGBoost)")
plt.gca().invert_yaxis()
plt.show()

