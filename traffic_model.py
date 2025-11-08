import pandas as pd
import numpy as np
import json
import time
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

df = pd.read_csv("Traffic_Volume.csv")
original = df.copy()

required_cols = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "date_time",
    "traffic_volume",
]

X = df.drop(columns=["traffic_volume"])
y = df["traffic_volume"]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

ct = ColumnTransformer(
    transformers=[
        ("cat", _make_ohe(), selector(dtype_include=object)),
        ("num", "passthrough", selector(dtype_exclude=object)),
    ],
    remainder="drop",
)

model = XGBRegressor(objective="reg:squarederror", random_state=42)
pipe = Pipeline(steps=[("prep", ct), ("model", model)])

mapie = MapieRegressor(estimator=pipe, n_jobs=-1, random_state=42)
start = time.time()
mapie.fit(train_X, train_y)
train_time = time.time() - start

alpha_default = 0.1
pred, pi = mapie.predict(test_X, alpha=alpha_default)
y_pred = pred.ravel()
y_pi = pi
r2 = r2_score(test_y, y_pred)
rmse = mean_squared_error(test_y, y_pred, squared=False)
residuals = (test_y.values - y_pred).tolist()
coverage = regression_coverage_score(test_y.values, y_pi[:, 0], y_pi[:, 1])

with open("traffic_xgb_mapie.pickle", "wb") as f:
    pickle.dump(mapie, f)

schema = {
    "required_columns": [
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "date_time",
    ],
    "target": "traffic_volume",
}
with open("traffic_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

sample_cols = schema["required_columns"] + [schema["target"]]
sample = original[sample_cols].head(5) if all(c in original.columns for c in sample_cols) else original.head(5)
sample.to_csv("traffic_sample_template.csv", index=False)

feature_names = mapie.estimator_.named_steps["prep"].get_feature_names_out()
importances = mapie.estimator_.named_steps["model"].feature_importances_.tolist()
feature_importance = dict(zip(feature_names.tolist(), importances))

eval_artifacts = {
    "train_time_sec": float(train_time),
    "alpha_default": float(alpha_default),
    "r2": float(r2),
    "rmse": float(rmse),
    "coverage_default_alpha": float(coverage),
    "test_pred": y_pred.tolist(),
    "test_pi_lower": y_pi[:, 0].tolist(),
    "test_pi_upper": y_pi[:, 1].tolist(),
    "test_y": test_y.tolist(),
    "residuals": residuals,
    "feature_importance": feature_importance,
}
with open("traffic_eval_artifacts.json", "w") as f:
    json.dump(eval_artifacts, f)

print(json.dumps({"train_time_sec": train_time, "r2": r2, "rmse": rmse, "coverage_default_alpha": coverage}, indent=2))
