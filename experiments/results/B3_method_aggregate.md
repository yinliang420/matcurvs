# B3 — Method-level aggregate table

Computed from `experiments/results/all_baselines.csv`. Cells show mean over the image set; n is the sample size the method covered in that stratum. Sorted by Inst-F1 descending within each stratum.

## ALL

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_M1_mock_v11 | ours_pipeline | 36 | 165.46 | 0.327 | 0.789 | 0.000 |
| ours_M1_v11b | ours_pipeline | 31 | 140.38 | 0.482 | 0.731 | 0.000 |
| vlm_qwen3.5-plus | vlm | 31 | 62.97 | 0.330 | 0.645 | 0.060 |
| ours_kmeans_m2f | ours_pipeline | 36 | 15.88 | 0.932 | 0.601 | 0.233 |
| vlm_gpt-5.4 | vlm | 31 | 114.70 | 0.207 | 0.571 | 0.091 |
| ours_kmeans_m2f_fallback | ours_pipeline | 36 | 25.29 | 0.852 | 0.471 | 0.076 |
| ours_M5_v11b | ours_pipeline | 31 | 110.33 | 0.819 | 0.460 | 0.000 |
| ours_M5_mock_v11 | ours_pipeline | 36 | 127.10 | 0.782 | 0.437 | 0.021 |
| ours_M2_mock_v11 | ours_pipeline | 36 | 113.81 | 0.865 | 0.437 | 0.013 |
| ours_M2_v11b | ours_pipeline | 31 | 105.01 | 0.855 | 0.435 | 0.000 |
| ours_M3_mock_v11 | ours_pipeline | 36 | 102.30 | 0.963 | 0.434 | 0.023 |
| ours_M3_v11b | ours_pipeline | 31 | 110.08 | 0.856 | 0.432 | 0.000 |
| ours_M6_v11b | ours_pipeline | 31 | 104.08 | 0.857 | 0.373 | 0.000 |
| ours_M4_v11b | ours_pipeline | 31 | 93.24 | 0.916 | 0.279 | 0.000 |
| ours_M6_mock_v11 | ours_pipeline | 36 | 101.07 | 0.942 | 0.265 | 0.013 |
| vlm_gemini-3.1-pro-preview | vlm | 31 | 81.66 | 0.097 | 0.265 | 0.064 |
| ours_M4_mock_v11 | ours_pipeline | 36 | 69.00 | 0.982 | 0.138 | 0.004 |
| vlm_claude-opus-4-6 | vlm | 31 | 76.48 | 0.034 | 0.086 | 0.004 |

## L1_single

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_M1_mock_v11 | ours_pipeline | 17 | 141.91 | 0.366 | 1.000 | 0.000 |
| vlm_qwen3.5-plus | vlm | 16 | 76.85 | 0.179 | 0.646 | 0.000 |
| ours_M1_v11b | ours_pipeline | 16 | 121.54 | 0.519 | 0.619 | 0.000 |
| ours_kmeans_m2f | ours_pipeline | 17 | 8.21 | 0.988 | 0.530 | 0.236 |
| vlm_gpt-5.4 | vlm | 16 | 111.20 | 0.000 | 0.500 | 0.000 |
| vlm_gemini-3.1-pro-preview | vlm | 16 | 94.66 | 0.125 | 0.330 | 0.032 |
| ours_kmeans_m2f_fallback | ours_pipeline | 17 | 18.82 | 0.914 | 0.329 | 0.050 |
| ours_M3_mock_v11 | ours_pipeline | 17 | 85.50 | 0.989 | 0.328 | 0.033 |
| ours_M5_v11b | ours_pipeline | 16 | 96.60 | 0.793 | 0.326 | 0.000 |
| ours_M3_v11b | ours_pipeline | 16 | 91.31 | 0.903 | 0.323 | 0.000 |
| ours_M6_v11b | ours_pipeline | 16 | 88.43 | 0.917 | 0.308 | 0.000 |
| ours_M2_v11b | ours_pipeline | 16 | 90.61 | 0.851 | 0.289 | 0.000 |
| ours_M2_mock_v11 | ours_pipeline | 17 | 92.01 | 0.933 | 0.286 | 0.000 |
| ours_M5_mock_v11 | ours_pipeline | 17 | 111.74 | 0.858 | 0.286 | 0.009 |
| ours_M6_mock_v11 | ours_pipeline | 17 | 90.93 | 0.901 | 0.227 | 0.000 |
| ours_M4_v11b | ours_pipeline | 16 | 78.02 | 0.920 | 0.186 | 0.000 |
| ours_M4_mock_v11 | ours_pipeline | 17 | 50.44 | 0.992 | 0.114 | 0.000 |
| vlm_claude-opus-4-6 | vlm | 16 | 107.81 | 0.003 | 0.104 | 0.000 |

## L2_multi_no_overlap

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_M1_v11b | ours_pipeline | 6 | 150.12 | 0.315 | 0.900 | 0.000 |
| vlm_qwen3.5-plus | vlm | 6 | 56.54 | 0.546 | 0.861 | 0.000 |
| vlm_gpt-5.4 | vlm | 6 | 71.38 | 0.411 | 0.778 | 0.150 |
| ours_kmeans_m2f | ours_pipeline | 6 | 22.44 | 0.937 | 0.658 | 0.319 |
| ours_M5_v11b | ours_pipeline | 6 | 97.34 | 0.736 | 0.611 | 0.000 |
| ours_M5_mock_v11 | ours_pipeline | 6 | 137.16 | 0.602 | 0.583 | 0.042 |
| ours_M2_v11b | ours_pipeline | 6 | 102.69 | 0.819 | 0.583 | 0.000 |
| ours_M2_mock_v11 | ours_pipeline | 6 | 110.82 | 0.738 | 0.583 | 0.039 |
| ours_M1_mock_v11 | ours_pipeline | 6 | 178.02 | 0.226 | 0.583 | 0.000 |
| ours_kmeans_m2f_fallback | ours_pipeline | 6 | 34.52 | 0.775 | 0.571 | 0.044 |
| ours_M3_v11b | ours_pipeline | 6 | 94.56 | 0.803 | 0.525 | 0.000 |
| ours_M3_mock_v11 | ours_pipeline | 6 | 70.08 | 0.888 | 0.512 | 0.019 |
| ours_M6_v11b | ours_pipeline | 6 | 82.81 | 0.803 | 0.476 | 0.000 |
| ours_M4_v11b | ours_pipeline | 6 | 93.30 | 0.929 | 0.346 | 0.000 |
| ours_M6_mock_v11 | ours_pipeline | 6 | 68.11 | 0.957 | 0.324 | 0.020 |
| ours_M4_mock_v11 | ours_pipeline | 6 | 81.77 | 0.996 | 0.172 | 0.005 |
| vlm_gemini-3.1-pro-preview | vlm | 6 | 120.40 | 0.000 | 0.167 | 0.079 |
| vlm_claude-opus-4-6 | vlm | 6 | nan | 0.000 | 0.000 | 0.000 |

## L3_multi_overlap

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_kmeans_m2f | ours_pipeline | 1 | 2.13 | 1.000 | 0.800 | 0.619 |
| ours_M1_mock_v11 | ours_pipeline | 1 | 148.39 | 0.241 | 0.667 | 0.000 |
| ours_kmeans_m2f_fallback | ours_pipeline | 1 | 14.55 | 0.853 | 0.667 | 0.274 |
| ours_M2_mock_v11 | ours_pipeline | 1 | 124.90 | 1.000 | 0.500 | 0.000 |
| ours_M3_mock_v11 | ours_pipeline | 1 | 74.45 | 1.000 | 0.500 | 0.140 |
| ours_M5_mock_v11 | ours_pipeline | 1 | 125.34 | 1.000 | 0.500 | 0.000 |
| ours_M4_mock_v11 | ours_pipeline | 1 | 72.07 | 1.000 | 0.400 | 0.000 |
| ours_M6_mock_v11 | ours_pipeline | 1 | 70.77 | 1.000 | 0.400 | 0.000 |

## L4_same_color_overlap

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_M1_v11b | ours_pipeline | 5 | 129.00 | 0.585 | 0.810 | 0.000 |
| ours_kmeans_m2f | ours_pipeline | 6 | 23.88 | 0.885 | 0.743 | 0.166 |
| vlm_gpt-5.4 | vlm | 5 | 217.06 | 0.390 | 0.700 | 0.381 |
| ours_kmeans_m2f_fallback | ours_pipeline | 6 | 28.77 | 0.834 | 0.688 | 0.107 |
| ours_M1_mock_v11 | ours_pipeline | 6 | 174.32 | 0.331 | 0.611 | 0.000 |
| ours_M5_v11b | ours_pipeline | 5 | 132.90 | 0.888 | 0.582 | 0.000 |
| ours_M2_v11b | ours_pipeline | 5 | 121.60 | 0.895 | 0.582 | 0.000 |
| ours_M5_mock_v11 | ours_pipeline | 6 | 127.19 | 0.619 | 0.568 | 0.000 |
| ours_M2_mock_v11 | ours_pipeline | 6 | 154.33 | 0.769 | 0.568 | 0.000 |
| ours_M3_v11b | ours_pipeline | 5 | 131.29 | 0.708 | 0.522 | 0.000 |
| ours_M3_mock_v11 | ours_pipeline | 6 | 124.86 | 0.962 | 0.500 | 0.000 |
| vlm_qwen3.5-plus | vlm | 5 | 12.40 | 0.393 | 0.400 | 0.371 |
| ours_M4_v11b | ours_pipeline | 5 | 107.82 | 0.873 | 0.364 | 0.000 |
| ours_M6_v11b | ours_pipeline | 5 | 108.90 | 0.738 | 0.317 | 0.000 |
| ours_M6_mock_v11 | ours_pipeline | 6 | 124.08 | 0.996 | 0.208 | 0.000 |
| vlm_gemini-3.1-pro-preview | vlm | 5 | 14.36 | 0.181 | 0.200 | 0.200 |
| ours_M4_mock_v11 | ours_pipeline | 6 | 81.98 | 0.968 | 0.103 | 0.000 |
| vlm_claude-opus-4-6 | vlm | 5 | nan | 0.000 | 0.000 | 0.000 |

## L5_dense_scatter

| method | family | n | DTW-Δ | CCR@0.1 | Inst-F1 | Curve-PQ |
|---|---|---|---|---|---|---|
| ours_M1_v11b | ours_pipeline | 4 | 215.39 | 0.454 | 0.831 | 0.000 |
| vlm_qwen3.5-plus | vlm | 4 | 59.74 | 0.530 | 0.625 | 0.000 |
| ours_M3_v11b | ours_pipeline | 4 | 181.90 | 0.934 | 0.617 | 0.000 |
| ours_M5_v11b | ours_pipeline | 4 | 156.53 | 0.963 | 0.617 | 0.000 |
| ours_M2_v11b | ours_pipeline | 4 | 145.37 | 0.876 | 0.617 | 0.000 |
| ours_M1_mock_v11 | ours_pipeline | 6 | 213.62 | 0.330 | 0.594 | 0.000 |
| ours_M3_mock_v11 | ours_pipeline | 6 | 164.21 | 0.958 | 0.578 | 0.000 |
| ours_M5_mock_v11 | ours_pipeline | 6 | 160.77 | 0.873 | 0.578 | 0.056 |
| ours_M2_mock_v11 | ours_pipeline | 6 | 136.18 | 0.875 | 0.578 | 0.037 |
| ours_kmeans_m2f | ours_pipeline | 6 | 25.37 | 0.805 | 0.571 | 0.140 |
| ours_M6_v11b | ours_pipeline | 4 | 192.57 | 0.851 | 0.550 | 0.000 |
| ours_kmeans_m2f_fallback | ours_pipeline | 6 | 32.73 | 0.770 | 0.523 | 0.115 |
| ours_M4_v11b | ours_pipeline | 4 | 135.84 | 0.934 | 0.443 | 0.000 |
| vlm_gpt-5.4 | vlm | 4 | 84.31 | 0.497 | 0.384 | 0.000 |
| ours_M6_mock_v11 | ours_pipeline | 6 | 144.82 | 0.979 | 0.350 | 0.055 |
| vlm_claude-opus-4-6 | vlm | 4 | 13.83 | 0.250 | 0.250 | 0.035 |
| vlm_gemini-3.1-pro-preview | vlm | 4 | 56.92 | 0.027 | 0.229 | 0.000 |
| ours_M4_mock_v11 | ours_pipeline | 6 | 95.29 | 0.953 | 0.166 | 0.021 |
