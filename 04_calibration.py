import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

RAW_CALIBRATION_FILE = "calibration_raw.csv"
MODEL_OUTPUT_FILE = "gaze_model.pkl"
METADATA_OUTPUT_FILE = "gaze_model_metadata.json"

def remove_point_outliers(df_point, feature_cols, iqr_factor=1.5):
    """Filters outlier frames for a single calibration target using IQR on feature norms."""
    if len(df_point) < 5:
        return df_point
    
    features = df_point[feature_cols].values
    center = np.median(features, axis=0)
    distances = np.linalg.norm(features - center, axis=1)
    
    q25, q75 = np.percentile(distances, [25, 75])
    iqr = q75 - q25
    cutoff = q75 + (iqr_factor * iqr)
    
    return df_point[distances <= cutoff]

def load_and_preprocess_calibration(csv_path):
    df = pd.read_csv(csv_path)
    
    # Identify feature columns (exclude target coordinates and metadata)
    ignore_cols = {'target_x', 'target_y', 'point_id', 'timestamp', 'frame_idx', 'accepted'}
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Create target point group identifier if not explicit
    if 'point_id' not in df.columns:
        df['point_id'] = df.groupby(['target_x', 'target_y']).ngroup()
    
    # Filter frame outliers per calibration target point
    filtered_dfs = []
    for point_id, group in df.groupby('point_id'):
        clean_group = remove_point_outliers(group, feature_cols)
        filtered_dfs.append(clean_group)
        
    clean_df = pd.concat(filtered_dfs, ignore_index=True)
    
    X = clean_df[feature_cols].values
    y = clean_df[['target_x', 'target_y']].values
    groups = clean_df['point_id'].values
    
    return X, y, groups, feature_cols, clean_df

def evaluate_models(X, y, groups):
    """Evaluates multiple regression architectures using spatial GroupKFold."""
    gkf = GroupKFold(n_splits=5)
    
    candidate_pipelines = {
        "Polynomial Ridge (Degree 2)": Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=10.0))
        ]),
        "Linear Ridge": Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        "MultiOutput SVR (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('svr', MultiOutputRegressor(SVR(C=10.0, epsilon=0.1)))
        ]),
        "Extra Trees Regressor": Pipeline([
            ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42))
        ])
    }
    
    results = {}
    best_score = float('inf')
    best_name = None
    best_pipeline = None
    
    print("\n--- Running GroupKFold Cross-Validation ---")
    for name, pipeline in candidate_pipelines.items():
        fold_errors = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            
            # Mean Euclidean Error in pixels
            euclidean_err = np.mean(np.linalg.norm(preds - y_val, axis=1))
            fold_errors.append(euclidean_err)
            
        mean_err = np.mean(fold_errors)
        results[name] = mean_err
        print(f"Model: {name:<30} | Group CV Mean Error: {mean_err:.2f} px")
        
        if mean_err < best_score:
            best_score = mean_err
            best_name = name
            best_pipeline = pipeline

    print(f"\nSelected Model: {best_name} ({best_score:.2f} px Group CV Error)")
    return best_pipeline, best_name, best_score

def main():
    if not os.path.exists(RAW_CALIBRATION_FILE):
        raise FileNotFoundError(f"Missing calibration data: {RAW_CALIBRATION_FILE}")
        
    X, y, groups, feature_cols, clean_df = load_and_preprocess_calibration(RAW_CALIBRATION_FILE)
    print(f"Loaded {len(clean_df)} valid frame samples across {len(np.unique(groups))} target points.")
    
    best_pipeline, best_model_name, cv_score = evaluate_models(X, y, groups)
    
    # Train selected model on the entire dataset
    best_pipeline.fit(X, y)
    
    # Save model artifact
    joblib.dump(best_pipeline, MODEL_OUTPUT_FILE)
    
    # Save metadata
    metadata = {
        "model_architecture": best_model_name,
        "sample_count": int(len(clean_df)),
        "unique_calibration_points": int(len(np.unique(groups))),
        "group_cv_euclidean_error_px": float(round(cv_score, 2)),
        "feature_columns": feature_cols,
        "participant_id": "child_session_default"
    }
    
    with open(METADATA_OUTPUT_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model saved to {MODEL_OUTPUT_FILE}")
    print(f"Metadata saved to {METADATA_OUTPUT_FILE}")

if __name__ == "__main__":
    main()