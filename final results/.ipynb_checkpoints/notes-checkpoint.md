# Experimental Notes

## Token count after merging

DeiT-B/16 produces 197 tokens (196 patch tokens + 1 CLS token) for 224×224 input.
With r tokens merged per layer across 12 layers:

| r  | Tokens at final layer | Calculation         |
|----|----------------------|---------------------|
| 4  | 149                  | 197 − (4 × 12)      |
| 8  | 101                  | 197 − (8 × 12)      |
| 16 | 5                    | 197 − (16 × 12)  ← likely too aggressive; expect significant accuracy drop |

r=16 reduces the sequence to just 5 tokens before the classification head, which
is expected to cause a large accuracy drop. Include in sweep for completeness but
note the degenerate behaviour in results.

## FlashAttention efficiency note

FlashAttention-2 is most efficient when sequence length N is a multiple of 64.
After ToMe merging, N will generally not be a multiple of 64 (e.g. 101, 149).
The kernel still produces correct results but may not reach peak theoretical
throughput. This is expected behaviour and should be noted when comparing against
published FlashAttention benchmarks that use power-of-2 or multiple-of-64 lengths.

## Baseline expected values

- DeiT-B/16 top-1 accuracy on ImageNet-1K val: **~81.8%**
- Reference: Touvron et al., "Training data-efficient image transformers &
  distillation through attention", ICML 2021.
- Checkpoint loaded via: `timm.create_model('deit_base_patch16_224', pretrained=True)`

## Preprocessing

Must match DeiT-B/16 training preprocessing exactly:
  1. `Resize(256)` — resize shorter edge to 256
  2. `CenterCrop(224)` — crop to 224×224
  3. `ToTensor()` — scale to [0, 1]
  4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet stats

Any deviation (e.g. using Resize(224) directly) will reduce measured accuracy and
make baseline numbers incomparable to the published ~81.8%.

## Statistical analysis

- α = 0.05 (pre-registered, two-sided tests)
- Throughput / memory: Welch's two-sample t-test (`scipy.stats.ttest_ind`, `equal_var=False`)
- Accuracy: two-proportion z-test (`statsmodels.stats.proportion.proportions_ztest`)
- 95% confidence intervals reported for all metrics
- 30 timed trials per condition (after 30 warm-up batches)
