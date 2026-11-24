Freight Rate Prediction Challenge

Solution for the Spotter AI Machine Learning Engineer assessment. The model predicts the posted freight rate for each shipment using a weighted blend of CatBoost and Extra Trees.

See Freight_Rate_Prediction_Report.pdf (or .docx) for the validation approach, data exploration, model selection and the December prediction chart.

## Repository structure

```
freight-rate-prediction/
├── data/                       # provided datasets
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── src/
│   └── train_blend.py          # training, validation and prediction
├── score.py                    # provided scorer (unmodified)
├── requirements.txt
├── validation_predictions.csv  # final submission (generated)
└── README.md
```


Requires Python 3.9 or later.

bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python -m pip install -r requirements.txt
Method
Features: distance, weight, pickup and delivery coordinates, market index, quote signal, calendar features (month, day of week, day of year), pickup city, delivery city and equipment.
Target: log(1 + posted_rate) during training, converted back to dollars for prediction.
Imputation: missing weight, market_index and quote_signal are filled with medians computed from the training rows only.
Models: CatBoost (categorical features handled natively) and Extra Trees (one-hot equipment, coordinates for location).
Blend: w × CatBoost + (1 − w) × ExtraTrees. The weight w is chosen on a 20% tuning holdout taken from the development data.
Validation

src/train_blend.py splits data/train_test.csv randomly into:

Development (80%), further split to select the blend weight on a 20% tuning holdout.
Test (20%), used only to report MAE, RMSE and R². The test set is not used for tuning.

The final models are refitted on all of train_test.csv before predicting the validation set and the December inputs.

Results (random 80:20 test split)
Model	MAE ($)	RMSE ($)	R²
CatBoost	97.61	530.99	0.8682
Extra Trees	101.64	531.58	0.8679
Blend (w = 0.55)	92.29	529.25	0.8690

A chronological check (train January–August, test September–October) gives a blend MAE of 124.78 and R² of 0.8255. The chronological result is the more realistic estimate of performance on the November–December validation period.

Run

Run from the repository root.

bash
# 1. Train the blend and write the predictions
python src/trained_model.py

# 2. Score the predictions (validates both files and creates the December chart)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

Outputs:

validation_predictions.csv: columns load_id,predicted_rate, 12,000 rows.
data/december_chart_inputs.csv: predicted_rate filled for each December date.
scorer_results/candidate_december.png: the December chart.

Runtime is several minutes, mostly from CatBoost training. Training runs three times: once for the tuning holdout, once for the test evaluation, and once for the final fit.

All random components use a fixed seed (SEED = 42), so results should match across runs on the same library versions. Library versions are pinned in requirements.txt.
