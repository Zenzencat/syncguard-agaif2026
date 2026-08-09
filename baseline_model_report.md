# SyncGuard baseline model -- RandomForest (attack vs. clean)

## Setup

- Features (23): fixType, gSpeed, hAcc, headAcc, numSV, pDOP, sAcc, vAcc, velD, velE, velN, pos_dev_m, n_sats_l1, snr_l1_mean, snr_l1_std, snr_l1_min, doppler_l1_mean, doppler_l1_std, pr_doppler_residual_mean, pr_doppler_residual_std, jam_ind_mean, agc_cnt_mean, noise_per_ms_mean
- `clock_drift_proxy_s` dropped (flat/uninformative).
- Scenario metadata (attack_type, rover_state, bands, timestamps, IDs) excluded from features -- not available to a deployed detector.
- Split: by recording (run_id), not by row, to avoid autocorrelation leakage. One dynamic + one stationary recording held out per attack_type.
- Train: 30562 rows / 16 recordings. Test: 14077 rows / 8 recordings.
- Class imbalance (77%/23% attack/clean) handled via `class_weight='balanced'`, not resampling.
- Model: RandomForestClassifier(n_estimators=300, class_weight='balanced'), median imputation for NaNs.

## Classification report (held-out recordings)

```
              precision    recall  f1-score   support

    clean(0)      0.759     0.644     0.697      3137
   attack(1)      0.902     0.941     0.921     10940

    accuracy                          0.875     14077
   macro avg      0.831     0.793     0.809     14077
weighted avg      0.870     0.875     0.871     14077

```

Confusion matrix [rows=true, cols=pred], order [clean, attack]:

```
[[ 2021  1116]
 [  642 10298]]
```

ROC-AUC: 0.916  
PR-AUC (average precision): 0.968

## Per-attack-type recall (held-out set, true attack rows only)

                    n_attack_rows    recall
attack_type                                
Jamming                    3498.0  0.870783
Meaconing                  1920.0  0.965104
Spoofing                   3120.0  0.972436
Spoofing + Jamming         2402.0  0.984596

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
