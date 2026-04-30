# Label/ y-lookup VLM Downstream Summary

**Model**: gemini-3.1-pro-preview  •  ±10% rel-error tolerance
**Images**: 100  •  **Questions**: 800

## Headline

- **acc_A (image-only baseline)** = 0.215  (n_valid=794)
- **acc_B (image + extracted curve)** = 0.388  (n_valid=800)
- **Δ = acc_B − acc_A** = **+0.172**
- **95% bootstrap CI on Δ** = [+0.106, +0.241]  (10000 resamples, cluster by image)
- Wilcoxon paired (one-sided, B>A) p = 1.991e-06  (n=100 paired imgs)

## Per-K breakdown

| K | n_imgs | n_q | acc_A | acc_B | Δ |
|---|---|---|---|---|---|
| 1 | 33 | 264 | 0.319 | 0.636 | +0.317 |
| 2 | 28 | 224 | 0.188 | 0.308 | +0.121 |
| 3 | 23 | 184 | 0.163 | 0.277 | +0.114 |
| 4 | 8 | 64 | 0.156 | 0.266 | +0.109 |
| 5 | 5 | 40 | 0.105 | 0.075 | -0.030 |
| 6 | 3 | 24 | 0.083 | 0.083 | +0.000 |

## Failure modes & honesty checks

- Parse-fail count: A=6, B=0  (rates 0.007 / 0.000)
- Missing responses: A=0, B=0
- Median latency: A=2.95s  B=4.84s
- Per-image B−A median = +0.125

### 10 worst image-level regressions (B−A)

| image_id | per-img Δ |
|---|---|
| 6ffe597ad3808f6f161f5194410a2e | -0.875 |
| 556ffa207fb7ef16f58d97a1c02748 | -0.625 |
| 3a81b9fc38a48abd05c768d0ca25c0 | -0.375 |
| 693ba352feaa8bebae6ee5b27fb9f5 | -0.375 |
| 00d117323855716750e5a0c3245c3d | -0.250 |
| 045c7254d8057d5eeff07cee4fba16 | -0.250 |
| 35a7e7a465dc5a3507eada9067bfb1 | -0.250 |
| 872dadbd93d95e4a06698350bebce2 | -0.250 |
| cd2a1b5212f03b2460a35c7ddf5c50 | -0.250 |
| ede31b540308cbff16adaded66df52 | -0.250 |

## Verdict

**B > A with 95% CI excluding 0** — extracted curve provides measurable scientific-reasoning value on numerical y-lookup.
