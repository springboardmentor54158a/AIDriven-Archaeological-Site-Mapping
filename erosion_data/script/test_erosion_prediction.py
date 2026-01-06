import joblib

model = joblib.load("erosion_rf_model.pkl")

# Example terrain point
new_point = [[10,0.25,120]]  # slope, veg_index, elevation
pred = model.predict(new_point)
risk = "Erosion-Prone" if pred[0]==1 else "Stable"
print("Predicted risk:", risk)
