import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import glob
from scipy import stats  # <--- Essential for T-test

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your FAST results
PATH_FAST = "/Users/kidus/Downloads/FAST/Results_finetune_only/FAST"

# Theoretical Chance Level (1/5 classes = 0.2)
CHANCE_LEVEL = 0.2 

def process_folder(folder_path, model_name="FAST", class_labels=None):
    print(f"\n--- Processing {model_name} in {folder_path} ---")
    
    # Matches files like "0-Tune.csv", "1-Tune.csv"
    search_path = os.path.join(folder_path, "*-Tune.csv")
        
    files = glob.glob(search_path)
    if not files:
        print(f"WARNING: No files found in {folder_path} matching pattern.")
        return None


    results = []
    all_cms = []  # To store all confusion matrices


    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            # Read CSV with header=None
            df = pd.read_csv(file_path, header=None)
            y_true = df.iloc[:, 0]
            y_pred = df.iloc[:, 1]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Compute and save confusion matrix plot
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            all_cms.append(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            import matplotlib.pyplot as plt
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: {filename}')
            plot_filename = os.path.splitext(filename)[0] + '_confusion_matrix.png'
            plot_path = os.path.join(folder_path, plot_filename)
            plt.savefig(plot_path)
            plt.close()

            results.append({
                'File': filename,
                'Accuracy': acc,
                'Precision': prec,
                'F1_Score': f1
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return pd.DataFrame(results), all_cms

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Define class labels (edit as needed)
class_labels = [0, 1, 2, 3, 4]  # Change if your classes are different

# 2. Process Data
df, all_cms = process_folder(PATH_FAST, class_labels=class_labels)

if df is not None and len(all_cms) > 0:
    accuracies = df['Accuracy'].values
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print("\n==========================================")
    print(f"MODEL PERFORMANCE SUMMARY (N={len(accuracies)})")
    print("==========================================")
    print(f"Mean Accuracy: {mean_acc:.4f} ({(mean_acc*100):.2f}%)")
    print(f"Std Deviation: {std_acc:.4f}")

    # 3. Average Confusion Matrix
    avg_cm = np.sum(np.array(all_cms), axis=0)
    import matplotlib.pyplot as plt
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm, display_labels=class_labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (All Subjects)')
    plt.savefig(os.path.join(PATH_FAST, 'average_confusion_matrix.png'))
    plt.close()

    # 4. ONE-SAMPLE T-TEST
    t_stat, p_val = stats.ttest_1samp(accuracies, popmean=CHANCE_LEVEL)
    if t_stat > 0:
        one_sided_p = p_val / 2
    else:
        one_sided_p = 1.0

    print("\n--- Statistical Significance (vs Chance 20%) ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {one_sided_p:.4e}")

    alpha = 0.05
    if one_sided_p < alpha:
        print(f"\nRESULT: SIGNIFICANT (p < {alpha})")
        print("Your model is performing significantly better than random guessing.")
    else:
        print(f"\nRESULT: NOT SIGNIFICANT (p >= {alpha})")
        print("Your model is not statistically distinguishable from random guessing.")

    print("==========================================\n")

    # Save results
    df.to_csv("FAST_Subject_Metrics.csv", index=False)
else:
    print("No data found.")