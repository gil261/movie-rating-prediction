# Movie Rating Prediction - Final Assignment Part 2

**Gil Hatiel & Liel Leone | Machine Learning Course**

## Model Description
ElasticNet regression model trained to predict IMDb average rating (`averageRating`)
before a movie's release, using features available prior to screening.

Best parameters: `alpha=0.0001`, `l1_ratio=0.1`
10-Fold CV RMSE: 1.1317 ± 0.0100

## How to Run

1. Place `dataset.csv` in the working directory
2. Run all cells in `notebook.ipynb` top to bottom
3. The final model is saved as `model.pkl`

## Test Evaluation
```python
df    = pd.read_csv("train.csv")
y     = df["averageRating"]
X     = prepare_data(df)
model = joblib.load("model.pkl")
y_pred = model.predict(X)
```

## Files
| File | Description |
|------|-------------|
| `notebook.ipynb` | Full pipeline: EDA, prepare_data, training, evaluation, analysis |
| `model.pkl` | Final ElasticNet model trained on all data |
| `report.pdf` | Full report (max 8 pages) |
| `README.md` | This file |
| `requirements.txt` | Required libraries and versions |

## Notes
- `title.crew.tsv` and `name.basics.tsv` (IMDb) must be in the working directory
  for director features to be computed. If unavailable, director features default to 0.
- Random seed: `random_state=42` throughout
- Python 3.10+
