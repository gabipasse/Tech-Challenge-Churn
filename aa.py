import joblib
a = joblib.load("artifacts/churn_pipeline.joblib")
print(type(a))
print(a.keys() if isinstance(a, dict) else a)