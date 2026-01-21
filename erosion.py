import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict(model,slope, dem, ndvi):
 
  
    X_new = pd.DataFrame({
        "Slope_Angle": [slope],
        "Elevation_m": [dem],
        "NDVI_Index": [ndvi]
    })

    # Fit a scaler on the training data and transform the input
    

    # Predict and return
    return model.predict(X_new)[0]
