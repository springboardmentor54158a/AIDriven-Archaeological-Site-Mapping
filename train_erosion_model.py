"""
Week 6: Train XGBoost or Random Forest Model for Erosion Prediction
Evaluate using RMSE and R2 Score
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features(df):
    """
    Prepare feature columns (exclude metadata)
    """
    exclude_cols = ['image_name', 'image_width', 'image_height', 
                    'erosion_probability', 'erosion_label', 'erosion_class']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def train_regression_model(X_train, y_train, X_val, y_val, model_type='xgboost'):
    """
    Train regression model to predict erosion probability
    """
    print(f"\nTraining {model_type} regression model...")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n{model_type.upper()} Regression Results:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Validation R²: {val_r2:.4f}")
    
    return model, {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred
    }

def train_classification_model(X_train, y_train, X_val, y_val, model_type='xgboost'):
    """
    Train classification model to predict erosion-prone vs stable
    """
    print(f"\nTraining {model_type} classification model...")
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n{model_type.upper()} Classification Results:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"\n  Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Stable', 'Erosion-prone']))
    
    return model, {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred
    }

def plot_feature_importance(model, feature_names, model_name, output_dir='erosion_results'):
    """
    Plot feature importance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importances = model.get_booster().get_score(importance_type='gain')
        # Convert to array
        importances = np.array([importances.get(f'f{i}', 0) for i in range(len(feature_names))])
    else:
        print("Cannot extract feature importance")
        return
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(10, 8))
    plt.title(f'{model_name} - Top 15 Feature Importance')
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
    plt.close()
    print(f"Feature importance plot saved to {output_dir}/{model_name}_feature_importance.png")

def plot_predictions(y_true, y_pred, model_name, output_dir='erosion_results'):
    """
    Plot predictions vs actual
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Erosion Probability')
    plt.ylabel('Predicted Erosion Probability')
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_predictions.png'))
    plt.close()

def main():
    """
    Main training pipeline
    """
    # Load labeled data
    if not os.path.exists('erosion_labeled_data.csv'):
        print("Labeled data not found. Please run prepare_erosion_labels.py first.")
        return
    
    df = pd.read_csv('erosion_labeled_data.csv')
    print(f"Loaded {len(df)} samples")
    
    # Prepare features
    feature_cols = prepare_features(df)
    X = df[feature_cols].values
    y_regression = df['erosion_probability'].values
    y_classification = df['erosion_label'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # Split data
    X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    results = {}
    
    # Train regression models
    for model_type in ['xgboost', 'random_forest']:
        # Regression
        reg_model, reg_results = train_regression_model(
            X_train, y_train_reg, X_val, y_val_reg, model_type
        )
        
        # Classification
        cls_model, cls_results = train_classification_model(
            X_train, y_train_cls, X_val, y_val_cls, model_type
        )
        
        # Save models
        joblib.dump(reg_model, f'erosion_{model_type}_regression.pkl')
        joblib.dump(cls_model, f'erosion_{model_type}_classification.pkl')
        
        # Feature importance
        plot_feature_importance(reg_model, feature_cols, f'{model_type}_regression')
        plot_feature_importance(cls_model, feature_cols, f'{model_type}_classification')
        
        # Plot predictions
        plot_predictions(y_val_reg, reg_results['y_val_pred'], f'{model_type}_regression')
        
        results[model_type] = {
            'regression': reg_results,
            'classification': cls_results
        }
    
    # Save results summary
    summary = {
        'XGBoost Regression': {
            'RMSE': results['xgboost']['regression']['val_rmse'],
            'R²': results['xgboost']['regression']['val_r2']
        },
        'Random Forest Regression': {
            'RMSE': results['random_forest']['regression']['val_rmse'],
            'R²': results['random_forest']['regression']['val_r2']
        },
        'XGBoost Classification': {
            'Accuracy': results['xgboost']['classification']['val_acc']
        },
        'Random Forest Classification': {
            'Accuracy': results['random_forest']['classification']['val_acc']
        }
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for model_name, metrics in summary.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save summary
    import json
    with open('erosion_model_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nModels saved:")
    print("  - erosion_xgboost_regression.pkl")
    print("  - erosion_xgboost_classification.pkl")
    print("  - erosion_random_forest_regression.pkl")
    print("  - erosion_random_forest_classification.pkl")
    print("\nResults saved to erosion_model_results.json")

if __name__ == "__main__":
    main()

