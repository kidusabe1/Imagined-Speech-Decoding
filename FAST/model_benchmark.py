import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
import glob
from scipy import stats  # <--- Essential for T-test

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your FAST results (per-subject folders produced by finetune CV)
PATH_FAST = "/home/kay/FAST/FAST/Results_finetune_official/FAST"
MODEL_NAME = "FAST"
PATH_GLOBAL = os.path.join(PATH_FAST, "global_test_predictions.csv")

# Theoretical Chance Level (1/5 classes = 0.2)
CHANCE_LEVEL = 0.2 

def process_subfolders(folder_path, model_name="FAST"):
    """Aggregate per-subject test predictions saved as sub-XX/test_predictions.csv."""
    print(f"\n--- Processing {model_name} in {folder_path} ---")

    sub_dirs = sorted(glob.glob(os.path.join(folder_path, "sub-*")))
    if not sub_dirs:
        print(f"WARNING: No subject folders found in {folder_path} matching sub-*/test_predictions.csv.")
        return None

    results = []

    for sub in sub_dirs:
        sid = os.path.basename(sub).replace('sub-', '')
        pred_file = os.path.join(sub, "test_predictions.csv")
        if not os.path.exists(pred_file):
            print(f"  Skipping {sub}: test_predictions.csv not found")
            continue
        try:
            df = pd.read_csv(pred_file)
            # Expect columns Predicted,True; fall back to positional if unnamed
            if {'Predicted', 'True'}.issubset(set(df.columns)):
                y_pred = df['Predicted']
                y_true = df['True']
            else:
                y_pred = df.iloc[:, 0]
                y_true = df.iloc[:, 1]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            results.append({
                'Subject': sid,
                'Accuracy': acc,
                'Precision': prec,
                'F1_Score': f1
            })
        except Exception as e:
            print(f"  Error reading {pred_file}: {e}")

    if not results:
        return None
    return pd.DataFrame(results)


def load_global_predictions(csv_path):
    """Load concatenated global predictions and compute metrics."""
    if not os.path.exists(csv_path):
        print(f"Global predictions file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Expect columns Predicted,True; fallback to positional
        if {'Predicted', 'True'}.issubset(set(df.columns)):
            y_pred = df['Predicted']
            y_true = df['True']
        else:
            y_pred = df.iloc[:, 0]
            y_true = df.iloc[:, 1]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            'Accuracy': acc,
            'Precision': prec,
            'F1_Score': f1
        }
    except Exception as e:
        print(f"Error reading global predictions: {e}")
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Process Data
df = process_subfolders(PATH_FAST, model_name=MODEL_NAME)
global_metrics = load_global_predictions(PATH_GLOBAL)

if df is not None:
    accuracies = df['Accuracy'].values
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n==========================================")
    print(f"MODEL PERFORMANCE SUMMARY (N={len(accuracies)})")
    print("==========================================")
    print(f"Mean Accuracy: {mean_acc:.4f} ({(mean_acc*100):.2f}%)")
    print(f"Std Deviation: {std_acc:.4f}")
    
    # 2. ONE-SAMPLE T-TEST
    # Tests if the population mean is significantly different from CHANCE_LEVEL
    t_stat, p_val = stats.ttest_1samp(accuracies, popmean=CHANCE_LEVEL)
    
    # We specifically want a "greater" test (is model > chance?), so we divide p by 2 
    # if t_stat is positive.
    if t_stat > 0:
        one_sided_p = p_val / 2
    else:
        one_sided_p = 1.0 # Model is worse than chance

    print("\n--- Statistical Significance (vs Chance 20%) ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {one_sided_p:.4e}")  # Scientific notation for very small numbers

    alpha = 0.05
    if one_sided_p < alpha:
        print(f"\nRESULT: SIGNIFICANT (p < {alpha})")
        print("Your model is performing significantly better than random guessing.")
    else:
        print(f"\nRESULT: NOT SIGNIFICANT (p >= {alpha})")
        print("Your model is not statistically distinguishable from random guessing.")
    
    print("==========================================\n")

    # Per-model summary row (subject-averaged)
    summary_rows = [{
        "Model": MODEL_NAME,
        "Acc_Mean": mean_acc,
        "Acc_Std": std_acc,
        "Prec_Mean": df['Precision'].mean(),
        "Prec_Std": df['Precision'].std(),
        "F1_Mean": df['F1_Score'].mean(),
        "F1_Std": df['F1_Score'].std(),
    }]

    # Add global (sample-weighted) metrics if available
    if global_metrics:
        summary_rows.append({
            "Model": f"{MODEL_NAME}_global",
            "Acc_Mean": global_metrics['Accuracy'],
            "Acc_Std": 0.0,
            "Prec_Mean": global_metrics['Precision'],
            "Prec_Std": 0.0,
            "F1_Mean": global_metrics['F1_Score'],
            "F1_Std": 0.0,
        })

    summary = pd.DataFrame(summary_rows)

    # Print summary in requested CSV style
    print("Model,Acc_Mean,Acc_Std,Prec_Mean,Prec_Std,F1_Mean,F1_Std")
    for _, row in summary.iterrows():
        print(f"{row['Model']},{row['Acc_Mean']},{row['Acc_Std']},{row['Prec_Mean']},{row['Prec_Std']},{row['F1_Mean']},{row['F1_Std']}")
    print()

    # Save results
    df.to_csv("/home/kay/FAST/FAST/Results_finetune_official/FAST_Subject_Metrics.csv", index=False)
    summary.to_csv("/home/kay/FAST/FAST/Results_finetune_official/Model_Summary.csv", index=False)
else:
    print("No data found.")