# SyncGuard improved model -- threshold tuning (jamming reweighting tried and rejected)

Targets the two limitations from `baseline_model_report.md` (clean recall 64.4%, jamming recall 87.1%). Full methodology, including what was tried and rejected on evidence, in the module docstring of `train_improved_model.py` and `ROBUSTNESS_NOTES.md`. Does not modify the baseline model, its report, or `train_baseline_model.py` (other than the additive `joblib.dump` at its end).

## What changed vs. the baseline model

**Only the decision threshold.** The underlying RandomForestClassifier has identical features, hyperparameters, and training data to the baseline -- this is the same model at a different operating point, not a retrained/reweighted one. Jamming-specific sample reweighting was tried first and rejected: measured directly against the real held-out TEST set, it made both clean recall and jamming recall *worse* than the baseline at every multiplier tried (see ROBUSTNESS_NOTES.md for the numbers). ROC-AUC/PR-AUC below are therefore expected to be identical to the baseline's (both are threshold-independent, computed on the same fitted model).

## Threshold selection (4-fold GroupKFold CV on training recordings only)

- Out-of-fold reference (threshold=0.5): clean_recall=0.472, jamming_recall=0.844.
- Chose threshold **0.52** -- the highest out-of-fold clean recall (0.498) among thresholds keeping out-of-fold jamming recall within 0.01 of the reference.
- The real held-out TEST set (same 8 recordings as the baseline) was used only for the one final evaluation below, at this already-chosen threshold.

## Classification report (held-out TEST, threshold=0.52)

```
              precision    recall  f1-score   support

    clean(0)      0.752     0.667     0.707      3137
   attack(1)      0.908     0.937     0.922     10940

    accuracy                          0.877     14077
   macro avg      0.830     0.802     0.815     14077
weighted avg      0.873     0.877     0.874     14077

```

Confusion matrix [rows=true, cols=pred], order [clean, attack]:

```
[[ 2093  1044]
 [  690 10250]]
```

ROC-AUC: 0.916 (baseline 0.916)  
PR-AUC: 0.968 (baseline 0.968)

## Per-attack-type recall (TEST, true attack rows only)

                    n_attack_rows    recall
attack_type                                
Jamming                    3498.0  0.858491
Meaconing                  1920.0  0.964583
Spoofing                   3120.0  0.971154
Spoofing + Jamming         2402.0  0.984596

Baseline: Jamming 0.871, Meaconing 0.965, Spoofing 0.972, Spoofing+Jamming 0.985.

## Clean-class recall

0.667 (baseline 0.644)

## Feature importances

```
agc_cnt_mean                0.134068
snr_l1_mean                 0.123277
noise_per_ms_mean           0.091191
jam_ind_mean                0.071361
snr_l1_std                  0.051909
velN                        0.050726
n_sats_l1                   0.047399
gSpeed                      0.047095
hAcc                        0.044299
vAcc                        0.037484
pr_doppler_residual_std     0.036591
doppler_l1_std              0.035316
sAcc                        0.032335
pDOP                        0.030830
doppler_l1_mean             0.028635
headAcc                     0.028626
pos_dev_m                   0.027056
snr_l1_min                  0.026654
numSV                       0.018183
velE                        0.015381
pr_doppler_residual_mean    0.013402
velD                        0.007591
fixType                     0.000591
```
