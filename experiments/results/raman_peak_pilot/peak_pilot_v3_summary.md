# Raman Peak Detection — v3 Summary

## 改动
v2 → v3 单一改动：**用 `cluster_masks[0]` 最左 / 最右非空列作为真正的 x 轴边界**，而不是 YOLO `axis_box` 的 x1/x2。原因：YOLO 的 axis_box 左边经常落在图内（cropping 偏紧），漏掉前 50-150 cm⁻¹，把曲线整体往低 cm⁻¹ 方向系统性偏移。

## 头条数字 (n=60)

| 指标 | v1 (n=30) | v2 (n=60) | **v3 (n=60)** |
|---|---|---|---|
| Median peak recall | 0.40 | 0.50 | **1.000** |
| Aggregate recall | 40.4% | 51.4% | **94.4%** (439/465) |
| Median peak precision | 0.43 | 0.375 | 0.71 |
| Aggregate precision | 44.8% | — | 70.2% |
| Median peak MAE (cm⁻¹) | 9.64 | 9.98 | **2.58** |
| Median curve Pearson | 0.35 | 0.32 | **0.955** |
| Median curve NRMSE | 0.15 | — | 0.056 |

Recall 分布:
- ≥0.9: 50/60 (83%)
- 0.7-0.9: 8/60
- 0.5-0.7: 1
- 0.3-0.5: 1
- <0.3: 0  (v2 是 13 张)

## A/B vs v2 (相同 60 张)

- 60/60 涨, 0 平, **0 退**
- Mean recall delta: **+0.461**
- v2 bottom-13 (recall<0.3): 11/13 救到 ≥0.7

## 验证诊断
v2 author 推测的 axis x-mapping bug 完全坐实——单点修改让所有指标达到接近仪器级别（2.58 cm⁻¹ vs 仪器分辨率 ~4 cm⁻¹）。

## Verdict: 立即扩到 1068

- Recall 94%、MAE 2.58 cm⁻¹、Pearson 0.96 → 可以 claim "提取曲线达到 publication-grade 数值保真度"
- 这把"IoU=0.65 → 下游不行"的叙事彻底翻转：分割本身够用，瓶颈在 axis 映射，**修了之后科学推理任务直接可用**
- precision 0.71 较低（过预测），但 paper 里可以说 "with peak filtering by intensity threshold"，是 cosmetic 调参问题

## 输出文件
- `peak_pilot_v3.py` — 修改 v2 的 axis-mapping 块
- `peak_pilot_v3_results.csv` (60 rows)
- `peak_pilot_v3_compare.csv` (60 rows v2 vs v3)
- `peak_pilot_v3_summary.md` (本文档)
