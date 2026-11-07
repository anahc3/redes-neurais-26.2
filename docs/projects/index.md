---
project: final
ai_use: "none"
---

# Final Project

Three deliverables, done as a team (same team and dataset across all three): Exploratory Data Analysis, then a Classification/Regression model, then a Generative pipeline.

| Deliverable | Deadline | Status |
|---|---|---|
| [Exploratory Data Analysis](https://thaisrs165.github.io/exploratory_data_analysis/) | Oct 25, 2026 | Done — hosted in the team repo |
| [Classification or Regression](https://thaisrs165.github.io/exploratory_data_analysis/projects/regression/) | Nov 13, 2026 | Done — hosted in the team repo |
| Generative | TBD | Not started |

## Exploratory Data Analysis

Submitted and hosted in the team repository, [Thaisrs165/exploratory_data_analysis](https://github.com/Thaisrs165/exploratory_data_analysis), published at **[thaisrs165.github.io/exploratory_data_analysis](https://thaisrs165.github.io/exploratory_data_analysis/)**. It covers two EDAs, one per task under consideration for the next deliverable:

- **Classification** — UCI Bank Marketing (`bank-full.csv`, target `y`)
- **Regression** — UCI Seoul Bike Sharing Demand (`SeoulBikeData.csv`, target `rented_bike_count`)

## Classification or Regression

Built **both** tracks, continuing directly from each EDA's pipeline and risk analysis. Submitted and hosted in the same team repository:

- **[Regression — Seoul Bike Sharing Demand](https://thaisrs165.github.io/exploratory_data_analysis/projects/regression/)**: a from-scratch-configured PyTorch MLP (22 → 64 → 32 → 1, predicting `sqrt(rented_bike_count)`) against trivial, linear (Ridge) and random-forest baselines.
    - **Test MAE 131.55 ± 3.07** bikes/hour (3 seeds) — a 53.2% reduction versus the required Ridge baseline (criterion: ≥ 15%), and edges out the optional random-forest baseline too
    - Honest negative result in the regularization search: dropout, weight decay and batch norm all *hurt* validation performance here — early stopping alone was enough
    - Error analysis lines up with the EDA's own findings: holiday hours are the worst-predicted subgroup (too few examples, as the EDA's risk list predicted), and permutation importance highlights the same top drivers (temperature, hour) the EDA's correlations found

- **[Classification — Bank Marketing](https://thaisrs165.github.io/exploratory_data_analysis/projects/classification/)**: a from-scratch-configured PyTorch MLP (51 → 32 → 16 → 1, logit output, `pos_weight`-handled imbalance) against trivial, linear (Logistic Regression) and random-forest baselines.
    - **Test PR-AUC 0.435 ± 0.003** (3 seeds) — a 6.4% relative improvement over the required Logistic Regression baseline (criterion was ≥ 15% — **not met**), and trails the optional random-forest baseline on PR-AUC
    - Opposite regularization finding from the regression track: dropout *helps* here, the one technique that did
    - Error analysis flags a possible data-collection confound (`day`) that the EDA had already warned about, and an overconfidence issue on the single strongest feature (`poutcome = success`)
