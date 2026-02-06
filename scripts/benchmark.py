#!/usr/bin/env python
"""
Benchmark script - Aggregate metrics from training results.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_subject_predictions(subject_folder):
    """Load test predictions for a single subject."""
    pred_path = os.path.join(subject_folder, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return None, None
    df = pd.read_csv(pred_path)
    if '# Predicted' in df.columns:
        return df['# Predicted'].values, df['True'].values
    return df.iloc[:, 0].values, df.iloc[:, 1].values


def load_global_predictions(results_dir):
    """Load global test predictions CSV."""
    global_path = os.path.join(results_dir, "global_test_predictions.csv")
    if not os.path.exists(global_path):
        return None, None
    df = pd.read_csv(global_path)
    if '# Predicted' in df.columns:
        return df['# Predicted'].values, df['True'].values
    return df.iloc[:, 0].values, df.iloc[:, 1].values


def process_results(results_dir, model_name="FAST"):
    """Process results directory and compute metrics."""
    
    model_folder = os.path.join(results_dir, model_name)
    if not os.path.exists(model_folder):
        print(f"Folder not found: {model_folder}")
        return None, None

    subjects = []
    metrics_list = []

    # Find all subject folders
    for item in sorted(os.listdir(model_folder)):
        if item.startswith("sub-"):
            subject_folder = os.path.join(model_folder, item)
            if os.path.isdir(subject_folder):
                sid = int(item.replace("sub-", ""))
                y_pred, y_true = load_subject_predictions(subject_folder)
                
                if y_pred is not None and y_true is not None:
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    precision = precision_score(y_true, y_pred, average='macro')
                    recall = recall_score(y_true, y_pred, average='macro')
                    
                    subjects.append(sid)
                    metrics_list.append({
                        'Subject': sid,
                        'Accuracy': acc,
                        'F1': f1,
                        'Precision': precision,
                        'Recall': recall,
                        'N_samples': len(y_true)
                    })

    if not metrics_list:
        print(f"No subject predictions found in {model_folder}")
        return None, None

    df_subjects = pd.DataFrame(metrics_list)
    
    # Compute global metrics from concatenated predictions
    y_pred_global, y_true_global = load_global_predictions(model_folder)
    
    if y_pred_global is not None and y_true_global is not None:
        global_acc = accuracy_score(y_true_global, y_pred_global)
        global_f1 = f1_score(y_true_global, y_pred_global, average='macro')
        global_precision = precision_score(y_true_global, y_pred_global, average='macro')
        global_recall = recall_score(y_true_global, y_pred_global, average='macro')
    else:
        # Fall back to mean of per-subject metrics
        global_acc = df_subjects['Accuracy'].mean()
        global_f1 = df_subjects['F1'].mean()
        global_precision = df_subjects['Precision'].mean()
        global_recall = df_subjects['Recall'].mean()

    summary = {
        'Model': model_name,
        'Acc_Mean': global_acc,
        'Acc_Std': df_subjects['Accuracy'].std(),
        'F1_Mean': global_f1,
        'F1_Std': df_subjects['F1'].std(),
        'Precision_Mean': global_precision,
        'Recall_Mean': global_recall,
        'N_subjects': len(df_subjects)
    }

    return df_subjects, summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark model results')
    parser.add_argument('--results_dir', type=str, default='results/finetune_official', 
                        help='Results directory')
    parser.add_argument('--model', type=str, default='FAST', help='Model name')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: results_dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    df_subjects, summary = process_results(args.results_dir, args.model)

    if df_subjects is not None:
        # Save per-subject metrics
        subj_path = os.path.join(output_dir, f"{args.model}_Subject_Metrics.csv")
        df_subjects.to_csv(subj_path, index=False)
        print(f"Saved subject metrics to: {subj_path}")

        # Save summary
        summary_path = os.path.join(output_dir, "Model_Summary.csv")
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved model summary to: {summary_path}")

        # Print summary
        print("\n" + "="*60)
        print(f"MODEL: {args.model}")
        print("="*60)
        print(f"Accuracy: {summary['Acc_Mean']:.4f} ± {summary['Acc_Std']:.4f}")
        print(f"F1 Score: {summary['F1_Mean']:.4f} ± {summary['F1_Std']:.4f}")
        print(f"Precision: {summary['Precision_Mean']:.4f}")
        print(f"Recall: {summary['Recall_Mean']:.4f}")
        print(f"N Subjects: {summary['N_subjects']}")
        print("="*60)


if __name__ == '__main__':
    main()
