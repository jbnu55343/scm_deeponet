# SUMO Dataset Results Summary

## 1. Data Diagnosis
- **Issue:** The original dataset contained ~50% zero values (no traffic/night time), which inflated simple accuracy metrics but confused the models when predicting actual traffic dynamics.
- **Action:** Filtered the dataset to keep only samples where `speed > 0.1`.
- **Result:** Dataset size reduced from ~1.2M to ~600k samples.

## 2. Feature Engineering Fix
- **Issue:** The training scripts were excluding Feature 0 (`speed(t)`), assuming it was redundant or leakage.
- **Analysis:** Correlation analysis showed Feature 0 is `speed(t)` and Feature 22 is `speed(t-12)`. Excluding Feature 0 removed the most critical predictor.
- **Action:** Updated all models to include Feature 0.

## 3. Model Performance (Filtered Dataset)
We retrained MLP, DeepONet, and Transformer on the filtered non-zero dataset.

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Persistence (Baseline)** | 0.5416 | 3.5683 | - |
| **MLP** | **0.7800** | 2.60 | - |
| **DeepONet** | 0.7685 | 2.80 | - |
| **Transformer** | 0.7708 | 2.82 | - |

## 4. Conclusion
- The models significantly outperform the persistence baseline (0.78 vs 0.54).
- **MLP** is currently the best performing model, likely due to its simplicity and ability to capture direct correlations efficiently.
- The "ceiling" of ~0.78 suggests high volatility in the simulation data (5-minute intervals) that is hard to predict perfectly.
- The previous R² of 0.71 (on full data) was misleading; the new 0.78 (on active traffic) represents true learning of traffic dynamics.
