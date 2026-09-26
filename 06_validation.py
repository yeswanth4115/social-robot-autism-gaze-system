import os
import joblib
import numpy as np
import pandas as pd

VALIDATION_FILE = "gaze_validation.csv"
MODEL_FILE = "gaze_model.pkl"

def analyze_validation(val_df, model):
    ignore_cols = {'target_x', 'target_y', 'point_id', 'timestamp', 'frame_idx', 'accepted'}
    feature_cols = [c for c in val_df.columns if c not in ignore_cols]
    
    X_val = val_df[feature_cols].values
    y_true = val_df[['target_x', 'target_y']].values
    
    y_pred = model.predict(X_val)
    
    # Calculate directional errors
    dx = y_pred[:, 0] - y_true[:, 0]
    dy = y_pred[:, 1] - y_true[:, 1]
    euclidean_errors = np.linalg.norm(y_pred - y_true, axis=1)
    
    val_df['pred_x'] = y_pred[:, 0]
    val_df['pred_y'] = y_pred[:, 1]
    val_df['err_x'] = dx
    val_df['err_y'] = dy
    val_df['err_euclidean'] = euclidean_errors
    
    # Overall Performance Metrics
    mean_err = np.mean(euclidean_errors)
    median_err = np.median(euclidean_errors)
    p95_err = np.percentile(euclidean_errors, 95)
    worst_err = np.max(euclidean_errors)
    
    mean_x_bias = np.mean(dx)
    mean_y_bias = np.mean(dy)
    
    print("=" * 60)
    print("        INDEPENDENT SPATIAL VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Validation Samples Evaluated : {len(val_df)}")
    print(f"Mean Euclidean Error               : {mean_err:.2f} px")
    print(f"Median Error                       : {median_err:.2f} px")
    print(f"95th Percentile Error              : {p95_err:.2f} px")
    print(f"Worst Frame Error                  : {worst_err:.2f} px")
    print(f"Overall Systematic Bias (dx, dy)   : ({mean_x_bias:+.2f} px, {mean_y_bias:+.2f} px)")
    
    # Accuracy Threshold Bins
    print("\n--- Accuracy Threshold Distribution ---")
    for threshold in [50, 100, 150, 200]:
        pct = (np.sum(euclidean_errors <= threshold) / len(euclidean_errors)) * 100
        print(f"Percentage within < {threshold:3d} px : {pct:6.2f}%")
        
    # Spatial Bias Breakdown by Screen Rows (Vertical Drift Analysis)
    print("\n--- Vertical Bias Breakdown by Target Y Row ---")
    print(f"{'Target Y':<10} | {'Mean Pred Y':<12} | {'Mean Y Error':<14} | {'Sample Count':<12}")
    print("-" * 56)
    for target_y, group in val_df.groupby('target_y'):
        avg_pred_y = group['pred_y'].mean()
        avg_err_y = group['err_y'].mean()
        print(f"{target_y:<10.0f} | {avg_pred_y:<12.1f} | {avg_err_y:<+14.1f} | {len(group):<12d}")

    # Spatial Bias Breakdown by Screen Columns (Horizontal Shift Analysis)
    print("\n--- Horizontal Bias Breakdown by Target X Column ---")
    print(f"{'Target X':<10} | {'Mean Pred X':<12} | {'Mean X Error':<14} | {'Sample Count':<12}")
    print("-" * 56)
    for target_x, group in val_df.groupby('target_x'):
        avg_pred_x = group['pred_x'].mean()
        avg_err_x = group['err_x'].mean()
        print(f"{target_x:<10.0f} | {avg_pred_x:<12.1f} | {avg_err_x:<+14.1f} | {len(group):<12d}")

def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not os.path.exists(VALIDATION_FILE):
        raise FileNotFoundError(f"Validation dataset not found: {VALIDATION_FILE}")
        
    model = joblib.load(MODEL_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)
    
    analyze_validation(val_df, model)

if __name__ == "__main__":
    main()