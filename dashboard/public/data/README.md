Drop exported results here after running `src/evaluation/export_dashboard_data.py` on Kaggle (or locally):

- `demand.json` — array of `{ timestamp, actual, predicted }`
- `metrics.json` — array of `{ model, mae, rmse, mape, trainingTimeSeconds }`
- `feature_importance.json` — array of `{ feature, importance }`

Any file that's missing falls back to mock data automatically (see `src/loadData.ts`).
