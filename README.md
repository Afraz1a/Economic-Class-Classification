# Economic Class Classification

A machine learning solution for predicting household economic class (lower / middle / upper) from individual-level survey data. Built for **CodeRush 2026 · ML Module**.
**Ranked in top 5 candidates**.

**Final Score: 0.7366 macro F1** on out-of-fold predictions.

---

## What makes this problem interesting

This isn't a standard classification task. Each household contains 3–8 people, and **all of them share a single label**. You have to predict the household's class from the combined data of its members. This is called **Multiple Instance Learning (MIL)**.

```
Household = { Dad, Mom, Son, Aunt, Gran }    →    one label: middle class
            (5 people, 21 features each)
```

The metric is **macro F1**, which weighs each class equally. Getting any single class wrong tanks the score, so every modeling choice has to consider all three classes.

---

## The numbers

| | |
|---|---|
| Training households | 3,360 |
| Total individuals | 16,776 |
| Test households | 400 |
| People per household | 3–8 (avg 5) |
| Engineered features | 333 |
| Models trained | 135 |
| Runtime | ~15 min on GPU |

**Class split:** 29% lower · 38% middle · 33% upper

---

## How it works

The pipeline runs in eight stages, splitting cleanly into preprocessing (deterministic) and modeling (where decisions live).

```
Load → Instance features → Bag aggregation → CV split → Train → OOF probs → Threshold tune → Submit
```

### 1. Drop leaky columns first

Eight columns get dropped before any other processing. The biggest offender is `interviewer_id` — when the same interviewer surveys households in both train and test, the model learns "interviewer #42 → middle class" instead of learning anything about wealth. This single fix is worth +0.02 to +0.05 macro F1.

Other drops: `survey_duration_mins`, `processing_flag`, `currency_code`, `interview_mode`, `poverty_line_usd`, `person_idx`, `annual_hours_est` (which equals 52 × `hours_per_week`).

### 2. Build per-person features

Each individual gets ~50 hand-crafted features across six categories: demographic, education, occupation, capital, work hours, family. The standout feature is `econ_proxy`:

```
econ_proxy = 10·occ_tier + edu_num + log(cap_gain) + 5·work_intensity
           + 3·is_married + 5·is_self_inc + 5·high_edu
```

A weighted score capturing wealth potential. When aggregated as `econ_proxy_max` (the household's wealthiest member's value), it becomes the model's #1 feature.

### 3. Encode domain knowledge

Trees can't infer that "Exec-managerial" outranks "Farming-fishing" from category names. Two lookup tables fix this:

- **`occ_tier`** — occupation ranked 0 (low-skill) to 3 (executive/professional)
- **`edu_score`** — education ranked 0 (preschool) to 16 (doctorate)

This monotonic encoding lets a single tree split capture most of the wealth signal.

### 4. Aggregate to one row per household

Each household's people get summarized into a 333-dimensional vector via:

- **Statistics** (×6 per feature): mean, max, min, std, sum, q75
- **Composition counts**: number of professionals, ratio married, etc.
- **Inequality measures**: Gini coefficient on capital, Shannon entropy on categories
- **Top-earner profile**: education, occupation, capital, hours of the wealthiest member ⭐
- **Category proportions**: marital status, occupation, race breakdowns

Top-earner features matter most because **household wealth is set by the wealthiest member, not the average**. A household with one doctor and four dependents looks middle-class on the mean but is upper-class on the max.

### 5. Train three diverse models

| Model | Tree growth | Why include it |
|---|---|---|
| LightGBM | Leaf-wise (deep, fast) | Fastest, often most accurate |
| XGBoost | Level-wise (balanced) | More conservative — different errors than LGB |
| CatBoost | Symmetric (oblivious) | Different again — adds ensemble diversity |

All three use gradient boosting with low learning rate (0.04), shallow depth (5–6), strong regularization, and early stopping at 60 rounds. **No Optuna** — conservative hand-picked hyperparameters favor stability over peak CV.

### 6. Cross-validate honestly

5 stratified folds × 3 random seeds = 15 train/val splits per algorithm = 135 total models. Multiple seeds decorrelate fold-level noise and give a more stable score estimate than a single 10-fold run.

### 7. Average, don't stack

Three GBDTs on the same features produce highly correlated outputs. Stacking them with a logistic regression meta-learner (what v2 did) just fits OOF noise. Simple element-wise averaging of the three probability matrices works better and gives variance reduction for free.

### 8. Tune thresholds post-hoc

This is the decisive trick. Instead of using `class_weight='balanced'` during training (which distorts probabilities and bakes in a fixed decision boundary), train on the natural distribution and tune class weights afterward.

```python
prediction(i) = argmax_k ( w_k · p_k(i) )
```

A grid search over 21³ = 9,261 combinations of `(w_lower, w_middle, w_upper)` finds the weights that maximize macro F1 directly.

**Best weights found:** `[1.175, 1.025, 0.875]` — boost lower, leave middle alone, dampen upper.

**Score boost:** 0.7260 → 0.7366 (+0.0106) with no retraining.

---

## Results

**Final OOF macro F1: 0.7366**

### Per-class F1

| Class | F1 |
|---|---|
| Lower | 0.66 |
| Middle | 0.69 |
| Upper | 0.86 |

### Confusion matrix

|  | Pred lower | Pred middle | Pred upper |
|---|---|---|---|
| **Actual lower** | 612 | 305 | 63 |
| **Actual middle** | 230 | 871 | 159 |
| **Actual upper** | 23 | 91 | 1,006 |

Upper class is easiest (clear capital signals). Lower class is hardest because it gets confused with middle (305 misclassified). Middle sits between both neighbors.

### Top features

`econ_proxy_max` dominates at 100% relative importance, followed by `top_cap` (87%), `capital_gain_max` (79%), `edu_x_occ_mean` (71%), and `log_cap_gain_max` (66%). Five of the top seven features are top-earner type — confirming the hypothesis that the household's wealthiest member determines its class.

---

## v2 vs v3

The first version reported CV ~0.75 but scored ~0.65 on Kaggle. A 10-point gap means the CV was lying. Each issue below contributed a small amount of optimism that compounded:

| Problem in v2 | Fix in v3 | Estimated gain |
|---|---|---|
| `interviewer_id` never dropped | Drop 8 noise columns up front | +0.02 to +0.05 |
| Optuna tuned on CV noise | Conservative fixed hyperparameters | +0.01 to +0.03 |
| `class_weight='balanced'` distorted probs | Post-hoc threshold tuning | +0.01 to +0.02 |
| LR stacking on correlated models | Simple averaging | +0.005 to +0.015 |
| 10 folds × 1 seed (high variance) | 5 folds × 3 seeds | +0.005 to +0.010 |
| `var < 1e-6` dropped binaries inconsistently | `var == 0.0` only | robustness |
| `multi_class` deprecation bug | Removed LR meta entirely | robustness |

**Net result:** v3 scores 0.7366 OOF with an expected LB of 0.71–0.73. The CV-LB gap shrank from 0.10 to ~0.02. A trustworthy 0.7366 beats a fake 0.75.

---

## Setup

```bash
git clone <your-repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, matplotlib, tqdm.

## Run

```bash
# Full pipeline
python main.py --train data/train.csv --test data/test.csv --output submission.csv

# Or step by step
python src/preprocess.py
python src/train.py
python src/tune_thresholds.py
python src/predict.py
```

Seeds are fixed in `config.py` (`[42, 137, 2024]`) for reproducibility.

---

## Project layout

```
.
├── data/                    raw and processed datasets
├── src/
│   ├── preprocess.py        clean + instance features + aggregation
│   ├── features.py          econ_proxy, gini, entropy
│   ├── lookup_tables.py     occ_tier, edu_score
│   ├── train.py             CV training loop
│   ├── ensemble.py          probability averaging
│   ├── tune_thresholds.py   grid search
│   └── predict.py           inference + submission
├── notebooks/               EDA and analysis
├── models/                  saved models
├── outputs/                 OOF probs, weights, submission.csv
├── config.py
├── main.py
└── requirements.txt
```

---

## Five takeaways

1. **Drop leaky columns first.** `interviewer_id` was worth more than any modeling improvement. Always look for shortcut features before tuning anything.
2. **Encode domain knowledge into features.** Hand-crafted ordinal tables and composite scores like `econ_proxy` give trees the right features to split on.
3. **Average correlated models, stack diverse ones.** Three GBDTs are too similar for stacking to help. Simple averaging works.
4. **Honest CV beats high CV.** Fewer folds with more seeds, no Optuna over-tuning, conservative hyperparameters. A 0.7366 you can trust beats a 0.75 you can't.
5. **Separate model training from decision-making.** Train on the natural distribution to get calibrated probabilities, then tune thresholds post-hoc to optimize the metric directly.

---

**Team Keyboard Warriors** · CodeRush 2026 · Kaggle ML Module
