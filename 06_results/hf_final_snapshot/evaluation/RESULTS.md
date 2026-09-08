# CardioMamba-Net — results

Generated 2026-09-07 13:31 UTC from `Shanmuk4622/cardiomamba-baselines-v2` and `Shanmuk4622/cardiomamba-net-v2`.

- runs analysed: **180**
- experiments: ['A_apnea', 'A_resting', 'A_valsalva', 'B_rva', 'C_all5', 'D_loso', 'F_cross:Apnea', 'F_cross:Resting', 'F_cross:Tilt-down', 'F_cross:Tilt-up', 'F_cross:Valsalva']
- variants: ['L10_transformer', 'L2_loss_only', 'L3_c1_only', 'L4_c1_c5', 'L5_no_wavelet', 'L6_no_ssm', 'L7_singletask', 'L8_no_film', 'L9_full', 'fpn', 'linknet', 'multireslinknet', 'unet']

- input completeness: **True**
- baseline reproduction gate passed: **False**
- evaluation validated: **False**

## Table 3 — RVA combined (headline)

| experiment   | variant                       |     MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   folds |   CC_t_paper |   CC_s_paper |   MAE_paper |   dCC_t |
|:-------------|:------------------------------|--------:|--------:|--------------:|--------------:|-----------------:|-----------------:|--------:|-------------:|-------------:|------------:|--------:|
| B_rva        | FPN                           | 0.15747 | 0.04042 |       53.2737 |       82.6544 |          0.51614 |          0.74268 |       5 |        59.63 |        69.53 |     0.14316 |   -6.36 |
| B_rva        | UNet                          | 0.16604 | 0.04333 |       28.7117 |       47.056  |          0.54815 |          0.81392 |       5 |        57.65 |        68.39 |     0.14798 |  -28.94 |
| B_rva        | LinkNet                       | 0.15591 | 0.0395  |       48.5501 |       79.3313 |          0.52041 |          0.73796 |       5 |        58.69 |        70.91 |     0.1478  |  -10.14 |
| B_rva        | MultiResLinkNet               | 0.16573 | 0.04269 |       44.6114 |       73.7015 |          0.5345  |          0.7623  |       5 |        61.86 |        79.96 |     0.14841 |  -17.25 |
| B_rva        | L2_loss_only                  | 0.15651 | 0.03792 |       52.8205 |       87.8362 |          0.87563 |          0.69332 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L3_c1_only                    | 0.16247 | 0.04046 |       59.6226 |       87.5538 |          0.89007 |          0.65083 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L4_c1_c5                      | 0.15422 | 0.03765 |       56.4364 |       87.7123 |          0.86209 |          0.77744 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L5_no_wavelet                 | 0.16201 | 0.03981 |       56.9078 |       89.0992 |          0.88718 |          0.71476 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L6_no_ssm                     | 0.16219 | 0.04162 |       51.5227 |       87.2138 |          0.89093 |          0.69975 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L7_singletask                 | 0.16281 | 0.04121 |       61.7394 |       88.8137 |          0.88346 |          0.82647 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | L8_no_film                    | 0.17155 | 0.04404 |       61.9847 |       89.1595 |          0.92399 |          0.69049 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | CardioMamba-Net (ours)        | 0.16487 | 0.04244 |       57.4644 |       87.4415 |          0.89921 |          0.75614 |       5 |       nan    |       nan    |   nan       |  nan    |
| B_rva        | CardioMamba-Net (Transformer) | 0.17013 | 0.04411 |       50.8874 |       87.5909 |          0.92301 |          0.73525 |       5 |       nan    |       nan    |   nan       |  nan    |

## Table 2 — per scenario

| experiment   | variant                |     MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   folds |   CC_t_paper |   CC_s_paper |   MAE_paper |   dCC_t |
|:-------------|:-----------------------|--------:|--------:|--------------:|--------------:|-----------------:|-----------------:|--------:|-------------:|-------------:|------------:|--------:|
| A_apnea      | FPN                    | 0.15469 | 0.03711 |       39.8153 |       75.6312 |          0.52456 |          0.78261 |       5 |        39.12 |        51.26 |     0.1531  |    0.7  |
| A_apnea      | UNet                   | 0.15939 | 0.0399  |       20.6502 |       44.4721 |          0.55189 |          0.92124 |       5 |        56.14 |        69.97 |     0.14406 |  -35.49 |
| A_apnea      | LinkNet                | 0.15062 | 0.0363  |       36.727  |       68.8462 |          0.51611 |          0.8503  |       5 |        56.22 |        70.35 |     0.14572 |  -19.49 |
| A_apnea      | MultiResLinkNet        | 0.15277 | 0.03657 |       41.7531 |       77.1238 |          0.52132 |          0.75752 |       5 |        55.33 |        74.66 |     0.14474 |  -13.58 |
| A_apnea      | CardioMamba-Net (ours) | 0.16453 | 0.0423  |       45.0714 |       82.5557 |          0.9266  |          0.9998  |       5 |       nan    |       nan    |   nan       |  nan    |
| A_resting    | FPN                    | 0.16019 | 0.04175 |       47.2623 |       80.3314 |          0.524   |          0.72313 |       5 |        58.37 |        71.38 |     0.14204 |  -11.11 |
| A_resting    | UNet                   | 0.16924 | 0.04455 |       20.4769 |       29.244  |          0.57079 |          0.92257 |       5 |        63.1  |        74.68 |     0.13872 |  -42.62 |
| A_resting    | LinkNet                | 0.17069 | 0.04694 |       45.106  |       77.2923 |          0.57268 |          0.79565 |       5 |        64.35 |        74.37 |     0.13588 |  -19.24 |
| A_resting    | MultiResLinkNet        | 0.16251 | 0.04119 |       49.9234 |       79.8625 |          0.53006 |          0.71038 |       5 |        66.1  |        82.44 |     0.13258 |  -16.18 |
| A_resting    | CardioMamba-Net (ours) | 0.17603 | 0.0469  |       56.658  |       87.5246 |          0.91832 |          0.80957 |       5 |       nan    |       nan    |   nan       |  nan    |
| A_valsalva   | FPN                    | 0.16249 | 0.04254 |       39.8628 |       76.3816 |          0.53763 |          0.78338 |       5 |        57.53 |        65.97 |     0.14985 |  -17.67 |
| A_valsalva   | UNet                   | 0.17832 | 0.04601 |       30.8099 |       53.557  |          0.57131 |          0.83828 |       5 |        58.38 |        68.79 |     0.15249 |  -27.57 |
| A_valsalva   | LinkNet                | 0.16266 | 0.04138 |       47.0882 |       79.7981 |          0.53798 |          0.76181 |       5 |        56.63 |        66.87 |     0.15087 |   -9.54 |
| A_valsalva   | MultiResLinkNet        | 0.16133 | 0.04158 |       51.4872 |       82.0556 |          0.53039 |          0.71381 |       5 |        60.14 |        77.05 |     0.15286 |   -8.65 |
| A_valsalva   | CardioMamba-Net (ours) | 0.16021 | 0.03984 |       51.4728 |       86.6917 |          0.86702 |          0.69925 |       5 |       nan    |       nan    |   nan       |  nan    |

## Table 3b — all five scenarios (new)

| experiment   | variant                |    MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   folds |
|:-------------|:-----------------------|-------:|--------:|--------------:|--------------:|-----------------:|-----------------:|--------:|
| C_all5       | CardioMamba-Net (ours) | 0.1866 | 0.05172 |       56.0237 |       86.7097 |          0.99146 |          0.72299 |       5 |

## Table 3c — leave-one-subject-out

| experiment   | variant                |     MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   folds |
|:-------------|:-----------------------|--------:|--------:|--------------:|--------------:|-----------------:|-----------------:|--------:|
| D_loso       | CardioMamba-Net (ours) | 0.17793 | 0.04798 |       54.9403 |       86.9048 |           0.9777 |          0.64524 |      30 |

## Table 3d — held-out scenario

| experiment        | variant                |     MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   folds |
|:------------------|:-----------------------|--------:|--------:|--------------:|--------------:|-----------------:|-----------------:|--------:|
| F_cross:Apnea     | CardioMamba-Net (ours) | 0.12597 | 0.0291  |       67.9895 |       91.069  |          0.73954 |          0.88357 |       1 |
| F_cross:Resting   | CardioMamba-Net (ours) | 0.10812 | 0.02539 |       75.1915 |       94.0593 |          0.60228 |          0.41955 |       1 |
| F_cross:Tilt-down | CardioMamba-Net (ours) | 0.10618 | 0.02503 |       73.3491 |       93.1007 |          0.63465 |          0.51592 |       1 |
| F_cross:Tilt-up   | CardioMamba-Net (ours) | 0.2092  | 0.0692  |       28.7019 |       81.6873 |          1.22774 |          0.65816 |       1 |
| F_cross:Valsalva  | CardioMamba-Net (ours) | 0.0842  | 0.01596 |       77.5761 |       94.5892 |          0.53389 |          0.39662 |       1 |

## Table 4 — R-peak detection

|                               |   accuracy |     F1 |   precision |   recall |       TP |        FP |       FN |   timing_err_ms |   missed_rate |
|:------------------------------|-----------:|-------:|------------:|---------:|---------:|----------:|---------:|----------------:|--------------:|
| FPN                           |     0.396  | 0.4782 |      0.5804 |   0.4181 |  8325.83 |  5500.67  | 11885.8  |         30.7726 |        0.5819 |
| UNet                          |     0.2691 | 0.3311 |      0.4636 |   0.2871 |  5528.33 |  3840.33  | 14683.3  |         35.4601 |        0.7129 |
| LinkNet                       |     0.223  | 0.3045 |      0.4048 |   0.255  |  5259.67 | 10354     | 14952    |         39.8003 |        0.745  |
| MultiResLinkNet               |     0.514  | 0.627  |      0.783  |   0.5545 | 11096.3  |  3230.17  |  9115.33 |         22.9601 |        0.4455 |
| L2_loss_only                  |     0.7662 | 0.8514 |      0.881  |   0.8332 | 16644.3  |  1731.83  |  3567.33 |         27.3872 |        0.1668 |
| L3_c1_only                    |     0.6786 | 0.7883 |      0.9324 |   0.7024 | 13881.8  |   770.833 |  6329.83 |         14.4965 |        0.2976 |
| L4_c1_c5                      |     0.6363 | 0.7488 |      0.8561 |   0.684  | 13497.3  |  1797     |  6714.33 |         20.0087 |        0.316  |
| L5_no_wavelet                 |     0.7105 | 0.8079 |      0.8879 |   0.7585 | 14962.7  |  1355.67  |  5249    |         17.3177 |        0.2415 |
| L6_no_ssm                     |     0.7292 | 0.8149 |      0.8797 |   0.7775 | 15189.2  |  1452.5   |  5022.5  |         16.9271 |        0.2225 |
| L7_singletask                 |     0.7702 | 0.8485 |      0.8918 |   0.8229 | 16271.3  |  1578.17  |  3940.33 |         15.4514 |        0.1771 |
| L8_no_film                    |     0.7725 | 0.8499 |      0.9016 |   0.8193 | 16256.5  |  1313.5   |  3955.17 |         15.3212 |        0.1807 |
| CardioMamba-Net (ours)        |     0.7178 | 0.8125 |      0.8944 |   0.7638 | 15138.3  |  1306     |  5073.33 |         16.4062 |        0.2362 |
| CardioMamba-Net (Transformer) |     0.6793 | 0.7799 |      0.861  |   0.7318 | 14526.3  |  1890.17  |  5685.33 |         18.8368 |        0.2682 |

## Table 5 — HR and HRV (real ms)

| variant                       | signal       |   mu_RR_ms |   sd_RR_ms |   mu_HR_bpm |   sd_HR_bpm |   RMSSD_ms |
|:------------------------------|:-------------|-----------:|-----------:|------------:|------------:|-----------:|
| FPN                           | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| FPN                           | predicted    |     986.59 |     297.59 |       69.46 |       22.28 |     405.01 |
| FPN                           | |error|      |     102.47 |     nan    |        9.34 |      nan    |     319.97 |
| UNet                          | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| UNet                          | predicted    |    1006.82 |     250.24 |       66.79 |       16.7  |     344.11 |
| UNet                          | |error|      |      88.37 |     nan    |        8.77 |      nan    |     263.52 |
| LinkNet                       | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| LinkNet                       | predicted    |     957.91 |     313.63 |       72.57 |       26.75 |     429.95 |
| LinkNet                       | |error|      |     111.5  |     nan    |       12.45 |      nan    |     344.91 |
| MultiResLinkNet               | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| MultiResLinkNet               | predicted    |     973.43 |     263.98 |       71.39 |       18.65 |     356.32 |
| MultiResLinkNet               | |error|      |     134.34 |     nan    |       13.61 |      nan    |     271.28 |
| L2_loss_only                  | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L2_loss_only                  | predicted    |     972.64 |     200.61 |       65.52 |       13.68 |     272.15 |
| L2_loss_only                  | |error|      |      46.27 |     nan    |        3.77 |      nan    |     187.11 |
| L3_c1_only                    | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L3_c1_only                    | predicted    |    1037.75 |     247.5  |       61.58 |       12.27 |     338.05 |
| L3_c1_only                    | |error|      |     100.07 |     nan    |        5.42 |      nan    |     253.01 |
| L4_c1_c5                      | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L4_c1_c5                      | predicted    |     990.61 |     256.08 |       66.29 |       18.85 |     347.15 |
| L4_c1_c5                      | |error|      |      64.33 |     nan    |        4.77 |      nan    |     262.11 |
| L5_no_wavelet                 | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L5_no_wavelet                 | predicted    |     993.16 |     227.39 |       64.83 |       15.68 |     301.87 |
| L5_no_wavelet                 | |error|      |      56.36 |     nan    |        3.76 |      nan    |     219.11 |
| L6_no_ssm                     | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L6_no_ssm                     | predicted    |     983.87 |     220.96 |       65.67 |       16.6  |     290.79 |
| L6_no_ssm                     | |error|      |      47.7  |     nan    |        3.69 |      nan    |     205.75 |
| L7_singletask                 | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L7_singletask                 | predicted    |     983.46 |     199.57 |       64.8  |       14.33 |     262.7  |
| L7_singletask                 | |error|      |      42.25 |     nan    |        2.98 |      nan    |     179.24 |
| L8_no_film                    | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| L8_no_film                    | predicted    |     988.84 |     192.89 |       64.04 |       12.73 |     253.2  |
| L8_no_film                    | |error|      |      45.03 |     nan    |        2.79 |      nan    |     169.01 |
| CardioMamba-Net (ours)        | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| CardioMamba-Net (ours)        | predicted    |     995.71 |     215.66 |       64.27 |       15.07 |     283.77 |
| CardioMamba-Net (ours)        | |error|      |      54.89 |     nan    |        3.2  |      nan    |     201.42 |
| CardioMamba-Net (Transformer) | ground truth |     943.8  |      94.46 |       65.67 |        7.45 |      85.04 |
| CardioMamba-Net (Transformer) | predicted    |     986.06 |     240.93 |       65.93 |       18.13 |     322.35 |
| CardioMamba-Net (Transformer) | |error|      |      61.56 |     nan    |        4.53 |      nan    |     238.13 |

## Table 6 — ablation ladder

|                                      |     MAE |     MSE |   CC_temporal |   CC_spectral |   RRMSE_temporal |   RRMSE_spectral |   peak_F1 |   MAE_mean_hr_bpm |   MAE_rmssd_ms |      params |   folds |   dCC_t |
|:-------------------------------------|--------:|--------:|--------------:|--------------:|-----------------:|-----------------:|----------:|------------------:|---------------:|------------:|--------:|--------:|
| 1. MultiResLinkNet + MSE (baseline)  | 0.16573 | 0.04269 |       44.6114 |       73.7015 |          0.5345  |          0.7623  |   0.6209  |          14.2823  |        263.951 | 9.21856e+06 |       5 |    0    |
| 2. + composite loss (C5)             | 0.15651 | 0.03792 |       52.8205 |       87.8362 |          0.87563 |          0.69332 |   0.8451  |           4.06323 |        185.827 | 9.21856e+06 |       5 |    8.21 |
| 3. + 8-channel input (C1)            | 0.16247 | 0.04046 |       59.6226 |       87.5538 |          0.89007 |          0.65083 |   0.77828 |           5.97986 |        253.342 | 9.22169e+06 |       5 |   15.01 |
| 4. + C1 + C5                         | 0.15422 | 0.03765 |       56.4364 |       87.7123 |          0.86209 |          0.77744 |   0.73822 |           5.81239 |        263.386 | 9.22169e+06 |       5 |   11.82 |
| 5. CardioMamba, no wavelet (-C2)     | 0.16201 | 0.03981 |       56.9078 |       89.0992 |          0.88718 |          0.71476 |   0.80698 |           4.2692  |        215.466 | 3.52249e+06 |       5 |   12.3  |
| 6. CardioMamba, no SSM (-C3)         | 0.16219 | 0.04162 |       51.5227 |       87.2138 |          0.89093 |          0.69975 |   0.80657 |           3.97339 |        206.815 | 3.09939e+06 |       5 |    6.91 |
| 7. CardioMamba, single-task (-C4)    | 0.16281 | 0.04121 |       61.7394 |       88.8137 |          0.88346 |          0.82647 |   0.83912 |           3.36991 |        184.076 | 4.0807e+06  |       5 |   17.13 |
| 8. CardioMamba, no FiLM              | 0.17155 | 0.04404 |       61.9847 |       89.1595 |          0.92399 |          0.69049 |   0.84009 |           3.11671 |        175.108 | 4.08819e+06 |       5 |   17.37 |
| 9. CardioMamba-Net (full)            | 0.16487 | 0.04244 |       57.4644 |       87.4415 |          0.89921 |          0.75614 |   0.80805 |           3.49774 |        200.83  | 4.09062e+06 |       5 |   12.85 |
| 10. Transformer bottleneck (control) | 0.17013 | 0.04411 |       50.8874 |       87.5909 |          0.92301 |          0.73525 |   0.77055 |           5.13629 |        239.586 | 4.5486e+06  |       5 |    6.28 |

## Table 7 — subject-paired Wilcoxon, Holm-corrected

| a                         | b                                    |   n_subjects |          p |   median_delta_cc_t |   p_holm | significant   |
|:--------------------------|:-------------------------------------|-------------:|-----------:|--------------------:|---------:|:--------------|
| 9. CardioMamba-Net (full) | 7. CardioMamba, single-task (-C4)    |           30 | 0.00218778 |            -2.2625  | 0.01969  | True          |
| 9. CardioMamba-Net (full) | 10. Transformer bottleneck (control) |           30 | 0.0120476  |             5.01185 | 0.096381 | False         |
| 9. CardioMamba-Net (full) | 1. MultiResLinkNet + MSE (baseline)  |           30 | 0.014538   |             3.20672 | 0.101766 | False         |
| 9. CardioMamba-Net (full) | 8. CardioMamba, no FiLM              |           30 | 0.0220989  |            -3.61593 | 0.132593 | False         |
| 9. CardioMamba-Net (full) | 6. CardioMamba, no SSM (-C3)         |           30 | 0.0549216  |             5.33819 | 0.274608 | False         |
| 9. CardioMamba-Net (full) | 2. + composite loss (C5)             |           30 | 0.0605652  |             3.84359 | 0.274608 | False         |
| 9. CardioMamba-Net (full) | 4. + C1 + C5                         |           30 | 0.626346   |             1.2722  | 1        | False         |
| 9. CardioMamba-Net (full) | 3. + 8-channel input (C1)            |           30 | 0.612006   |             0.31199 | 1        | False         |
| 9. CardioMamba-Net (full) | 5. CardioMamba, no wavelet (-C2)     |           30 | 0.776569   |            -0.52298 | 1        | False         |

## Table 8 — budget

|                                   |   M_params |   CC_temporal |   CC_per_Mparam |
|:----------------------------------|-----------:|--------------:|----------------:|
| 8. CardioMamba, no FiLM           |      4.088 |       61.9847 |           15.16 |
| 7. CardioMamba, single-task (-C4) |      4.081 |       61.7394 |           15.13 |
| 3. + 8-channel input (C1)         |      9.222 |       59.6226 |            6.47 |
| CardioMamba-Net (ours)            |      4.091 |       57.4645 |           14.05 |
| 5. CardioMamba, no wavelet (-C2)  |      3.522 |       56.9078 |           16.16 |
| 4. + C1 + C5                      |      9.222 |       56.4364 |            6.12 |
| FPN                               |      2.237 |       53.2737 |           23.81 |
| 2. + composite loss (C5)          |      9.219 |       52.8205 |            5.73 |
| 6. CardioMamba, no SSM (-C3)      |      3.099 |       51.5227 |           16.63 |
| CardioMamba-Net (Transformer)     |      4.549 |       50.8874 |           11.19 |
| LinkNet                           |      2.775 |       48.5501 |           17.5  |
| MultiResLinkNet                   |      9.219 |       44.6114 |            4.84 |
| UNet                              |     12.211 |       28.7117 |            2.35 |

## Table 9 — robustness

| variant         | corruption      |   level | channel   |   n | comparison_scale   |      MAE |   MAE_std |       MSE |   MSE_std |   CC_temporal |   CC_temporal_std |   CC_spectral |   CC_spectral_std |   RRMSE_temporal |   RRMSE_temporal_std |   RRMSE_spectral |   RRMSE_spectral_std |        R2 |   R2_std |
|:----------------|:----------------|--------:|:----------|----:|:-------------------|---------:|----------:|----------:|----------:|--------------:|------------------:|--------------:|------------------:|-----------------:|---------------------:|-----------------:|---------------------:|----------:|---------:|
| L9_full         | clean           |    0    |           | 800 | [0,1]              | 0.140184 | 0.0404183 | 0.0303577 | 0.013938  |    63.0083    |          20.9998  |       86.7829 |           11.4656 |         0.481668 |            0.244049  |         0.658411 |           0.754461   | -0.443967 | 1.14121  |
| L9_full         | awgn            |   12    |           | 800 | [0,1]              | 0.14009  | 0.0408228 | 0.0306094 | 0.0139726 |    60.5152    |          21.7206  |       86.9543 |           11.2201 |         0.481869 |            0.237592  |         0.657061 |           0.678237   | -0.451233 | 1.13521  |
| L9_full         | awgn            |    6    |           | 800 | [0,1]              | 0.141118 | 0.0389206 | 0.0312011 | 0.0133057 |    55.1214    |          22.623   |       86.7851 |           11.1904 |         0.484655 |            0.227544  |         0.639751 |           0.385816   | -0.467634 | 1.05347  |
| L9_full         | awgn            |    3    |           | 800 | [0,1]              | 0.146221 | 0.0355837 | 0.0331276 | 0.0124437 |    49.2238    |          23.626   |       86.328  |           11.5784 |         0.498986 |            0.221359  |         0.638989 |           0.199517   | -0.540941 | 0.934991 |
| L9_full         | awgn            |    0    |           | 800 | [0,1]              | 0.153334 | 0.0315205 | 0.0363348 | 0.0111982 |    40.8275    |          22.6424  |       84.7844 |           13.1339 |         0.523879 |            0.222883  |         0.669477 |           0.166529   | -0.671415 | 0.813837 |
| L9_full         | awgn            |   -3    |           | 800 | [0,1]              | 0.160457 | 0.0296223 | 0.0406398 | 0.0113333 |    29.4466    |          19.8153  |       81.3987 |           14.9546 |         0.550565 |            0.21728   |         0.73397  |           0.135978   | -0.842663 | 0.775251 |
| L9_full         | motion_drift    |    0.25 |           | 800 | [0,1]              | 0.140356 | 0.0600492 | 0.0319569 | 0.0211379 |    62.1403    |          21.7029  |       87.7044 |           12.1855 |         0.501741 |            0.309938  |         0.653955 |           0.737794   | -0.543216 | 1.35212  |
| L9_full         | motion_drift    |    0.5  |           | 800 | [0,1]              | 0.162512 | 0.0968406 | 0.0443471 | 0.0437466 |    59.019     |          23.1954  |       87.7717 |           12.8544 |         0.582924 |            0.442134  |         0.667632 |           0.704077   | -1.19836  | 2.46433  |
| L9_full         | channel_dropout |    1    | I         | 800 | [0,1]              | 0.141469 | 0.0421089 | 0.0305224 | 0.0147375 |    63.7885    |          22.0628  |       87.7245 |           10.9998 |         0.482575 |            0.247343  |         0.575651 |           0.412621   | -0.433702 | 1.04489  |
| L9_full         | channel_dropout |    1    | Q         | 800 | [0,1]              | 0.157123 | 0.0412006 | 0.0341513 | 0.014753  |    67.2671    |          23.1072  |       88.2752 |           11.9783 |         0.510166 |            0.263487  |         0.553408 |           0.378559   | -0.633262 | 1.45763  |
| L9_full         | channel_dropout |    1    | phi       | 800 | [0,1]              | 0.146612 | 0.0530226 | 0.0335282 | 0.0199052 |    62.4129    |          22.2769  |       86.7032 |           11.5945 |         0.512171 |            0.324435  |         0.617043 |           0.378681   | -0.614222 | 1.5066   |
| L9_full         | channel_dropout |    1    | dy        | 800 | [0,1]              | 0.134628 | 0.0373778 | 0.0285997 | 0.0128807 |    64.1237    |          19.978   |       87.9685 |           10.5968 |         0.462443 |            0.21248   |         0.676171 |           0.835499   | -0.357862 | 1.09225  |
| L9_full         | channel_dropout |    1    | vel       | 800 | [0,1]              | 0.157821 | 0.0453129 | 0.0368868 | 0.0146031 |    62.9687    |          24.1173  |       85.7819 |           12.7224 |         0.517723 |            0.216444  |         0.768392 |           0.927738   | -0.687858 | 1.00135  |
| L9_full         | channel_dropout |    1    | acc       | 800 | [0,1]              | 0.134917 | 0.0391999 | 0.0297953 | 0.0131271 |    58.7145    |          24.3208  |       86.12   |           12.8201 |         0.483068 |            0.255     |         0.723042 |           1.01534    | -0.408829 | 1.00108  |
| L9_full         | channel_dropout |    1    | amp       | 800 | [0,1]              | 0.158044 | 0.0499144 | 0.0364712 | 0.0159986 |    62.5665    |          20.5739  |       87.2554 |           11.3342 |         0.509597 |            0.220661  |         0.642862 |           0.72895    | -0.710284 | 1.2817   |
| L9_full         | channel_dropout |    1    | cardiac   | 800 | [0,1]              | 0.135101 | 0.0547369 | 0.0305481 | 0.0199292 |    56.3348    |          18.9199  |       85.7752 |           11.1445 |         0.500696 |            0.329259  |         0.679208 |           0.235626   | -0.543988 | 2.23107  |
| multireslinknet | clean           |    0    |           | 800 | [0,1]              | 0.175044 | 0.0524329 | 0.0429106 | 0.0200473 |    57.4862    |          25.0397  |       81.878  |           25.5321 |         0.577476 |            0.334556  |         0.648472 |           0.429154   | -1.05045  | 1.53526  |
| multireslinknet | awgn            |   12    |           | 800 | [0,1]              | 0.15393  | 0.0407579 | 0.0424317 | 0.0157996 |     8.92459   |          15.7825  |       74.7872 |           18.865  |         0.55418  |            0.205963  |         0.833681 |           0.111359   | -0.890332 | 0.867178 |
| multireslinknet | awgn            |    6    |           | 800 | [0,1]              | 0.168411 | 0.0572731 | 0.0498684 | 0.027963  |     3.6863    |          10.0832  |       66.8992 |           26.7499 |         0.584857 |            0.209411  |         0.923403 |           0.0740536  | -1.22147  | 1.49487  |
| multireslinknet | awgn            |    3    |           | 800 | [0,1]              | 0.180985 | 0.0681357 | 0.0563218 | 0.0360661 |     2.0035    |           8.31723 |       61.2986 |           29.9954 |         0.607095 |            0.207137  |         0.944224 |           0.057646   | -1.52302  | 1.95725  |
| multireslinknet | awgn            |    0    |           | 800 | [0,1]              | 0.194993 | 0.0808452 | 0.0643643 | 0.0444407 |     1.54654   |           6.701   |       53.8875 |           30.1123 |         0.630531 |            0.198743  |         0.952504 |           0.0404826  | -1.89722  | 2.48232  |
| multireslinknet | awgn            |   -3    |           | 800 | [0,1]              | 0.21061  | 0.0937948 | 0.0741513 | 0.0523352 |     1.15016   |           5.81352 |       44.0678 |           28.6051 |         0.659897 |            0.190853  |         0.956301 |           0.0317994  | -2.34456  | 2.98745  |
| multireslinknet | motion_drift    |    0.25 |           | 800 | [0,1]              | 0.1727   | 0.0517962 | 0.0423832 | 0.0197519 |    54.6829    |          23.614   |       81.5759 |           24.1917 |         0.578496 |            0.338377  |         0.647602 |           0.333468   | -1.03493  | 1.57547  |
| multireslinknet | motion_drift    |    0.5  |           | 800 | [0,1]              | 0.167338 | 0.0545084 | 0.0415971 | 0.0199253 |    47.5504    |          22.6784  |       78.121  |           22.1545 |         0.583881 |            0.357649  |         0.703156 |           0.267788   | -1.01233  | 1.61529  |
| multireslinknet | channel_dropout |    1    | dy        | 800 | [0,1]              | 0.187085 | 0.0693045 | 0.0595072 | 0.029806  |    -0.0188151 |           6.50482 |       27.4378 |           11.7061 |         0.588088 |            0.0861945 |         0.989107 |           0.00571855 | -1.53562  | 1.46652  |

## Figures

- `figures/fig01_ours_vs_published.png`
- `figures/fig02_ablation.png`
- `figures/fig03_bland_altman_hr.png`
- `figures/fig04_bland_altman_rmssd.png`
- `figures/fig05_distribution.png`
- `figures/fig06_per_subject.png`
- `figures/fig07_qualitative.png`
- `figures/fig08_budget.png`
- `figures/fig09_robustness.png`

## Reading notes for the manuscript

- Our splits are strictly subject-wise with non-overlapping test windows. The baseline's Table 1 counts carry the 50 % overlap and are split 80/20, which permits overlapping windows across train and test. Baseline rows landing below their published values is the expected consequence of removing that, not a weaker implementation.
- Correlations are reported x100 throughout, matching the baseline's tables.
- MAE and MSE are standardized to the paper's [0,1] waveform scale. The lossless all_runs.csv and per-window data retain MAE_recorded/MSE_recorded and the original scale.
- Temporal RRMSE is retained exactly as recorded. Because it depends on the target's DC offset, it must not be compared directly between NB03 [0,1] and NB04 [-1,1] runs.
- mu_RR is in genuine milliseconds. The baseline's Table 5 reports 126 ms alongside 62 bpm, which is arithmetically impossible; 126 samples at 128 Hz is 0.98 s.
- Significance uses subject-paired Wilcoxon tests for predeclared full-model comparisons with Holm correction.