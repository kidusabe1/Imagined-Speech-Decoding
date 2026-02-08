# Train/Val/Test Split Analysis: GitHub FAST vs Our Repository

## GitHub FAST Repository (Jiang-Muyun/FAST)

### Overview
- Dataset: BCI Competition 2020 Track #3 (15 subjects, 5 classes)
- Subject-independent evaluation (each subject processed independently)

### Splitting Strategy
1. **Preprocessing**: Official Training set + Validation set are **merged** per subject
2. **CV**: `KFold(n_splits=5, shuffle=False)` on the combined data
3. **No separate validation set** — trains for fixed 200 epochs, no early stopping
4. **No held-out test set** — the KFold held-out fold IS the evaluation
5. **Batch size**: Full training set as one batch (`batch_size=len(x_train)`)

### Per Fold
- 80% train / 20% test (KFold "test" = evaluation)
- No model selection based on validation accuracy

---

## Our Repository

### FAST Training (`scripts/train_fast.py`)
1. **Preprocessing**: Official Training set + Validation set merged per subject
2. **CV**: `KFold(n_splits=5, shuffle=True, random_state=42)`
3. **Validation**: KFold held-out fold used as validation for model selection (best `val_acc`)
4. **Test set**: Official BCI Competition test set kept completely separate, used only for final evaluation
5. **Batch size**: 64 (mini-batches)

### TSception Training (`scripts/train_tsception.py`)
1. **CV**: `KFold(n_splits=5, shuffle=False)`
2. **Internal validation**: 15% split from training portion (`train_test_split(test_size=0.15, stratify=True)`)
3. **Test set**: Official test set evaluated after identifying best fold
4. **Batch size**: 32

---

## Key Differences Summary

| Aspect | GitHub FAST | Our FAST | Our TSception |
|--------|-------------|----------|---------------|
| KFold shuffle | False | True (seed=42) | False |
| Validation set | None | KFold held-out fold | 15% internal split |
| Test set | None (KFold test = eval) | Official test set | Official test set |
| Batch size | Full batch | 64 | 32 |
| Early stopping | No (fixed 200 epochs) | Yes (best val_acc) | Yes |
| Model selection | None | Best val_acc checkpoint | Best fold model |

## Conclusion
Our repo has a more rigorous evaluation setup with proper validation and a held-out test set. The GitHub repo's reported "test" accuracy is really KFold held-out fold accuracy, which can overestimate generalization.
