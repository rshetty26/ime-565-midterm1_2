import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

st.set_page_config(page_title="Traffic Volume Prediction", layout="wide")
st.title("Traffic Volume Prediction (XGBoost + MAPIE)")
st.image("traffic_image.gif")

DATA_PATH = "Traffic_Volume.csv"
PICKLE_PATH = "traffic_model.pickle"

REQUIRED_COLS = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "date_time"]
ALL_COLS_WITH_TARGET = ["traffic_volume"] + REQUIRED_COLS

CATEGORICAL_COLS = ["holiday", "weather_main"]

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_base_model(pickle_path: str) -> BaseEstimator:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def ensure_required_columns(df: pd.DataFrame, need_target: bool = False) -> tuple[bool, list]:
    cols = ALL_COLS_WITH_TARGET if need_target else REQUIRED_COLS
    missing = [c for c in cols if c not in df.columns]
    return (len(missing) == 0, missing)

def extract_underlying_estimator(model) -> BaseEstimator | None:
    if isinstance(model, MapieRegressor):
        base = getattr(model, "estimator", None)
        if base is not None:
            return base
    return model

def safe_feature_importances(model: BaseEstimator, X_fit_columns: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """
    Attempts to read feature_importances_ from underlying model or last pipeline step.
    Falls back to zeros (with names) if unavailable.
    """
    try:
        base = extract_underlying_estimator(model)
        feature_names = X_fit_columns or []
        importances = None

        if isinstance(base, Pipeline):
            if hasattr(base[-1], "feature_importances_"):
                importances = base[-1].feature_importances_
            # Try to fetch feature names from transformers (if any)
            for _, step in base.steps[:-1][::-1]:
                if hasattr(step, "get_feature_names_out"):
                    try:
                        feature_names = list(step.get_feature_names_out())
                        break
                    except Exception:
                        pass
        else:
            if hasattr(base, "feature_importances_"):
                importances = base.feature_importances_

        if importances is not None:
            if not feature_names or len(feature_names) != len(importances):
                feature_names = [f"f{i}" for i in range(len(importances))]
            return np.array(importances), feature_names
    except Exception:
        pass

    if X_fit_columns:
        return np.zeros(len(X_fit_columns)), X_fit_columns
    return np.array([]), []

def preprocess_raw(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, robust preprocessing for Traffic Volume:
      - Parse date_time -> hour/dayofweek/month/is_weekend + cyclical hour
      - Keep raw weather features as-is (in same units as dataset)
      - Do not convert temp unit (dataset is Kelvin)
      - Return frame that still contains 'holiday' and 'weather_main' for one-hot later
    """
    df = df_in.copy()
    # Parse datetime
    # The dataset looks like  "10/2/12 9:00" => "%m/%d/%y %H:%M"
    # If user CSV uses ISO timestamps, pandas will parse automatically as well.
    try:
        df["date_time"] = pd.to_datetime(df["date_time"], format="%m/%d/%y %H:%M")
    except Exception:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Drop the raw timestamp (models should not one-hot the timestamp)
    df = df.drop(columns=["date_time"])

    # Keep holiday/weather_main as categoricals for one-hot later
    return df

def build_dummy_template(df_full: pd.DataFrame) -> list[str]:
    """
    Build a template of expected dummy columns from the training dataframe.
    We preprocess first, then dummify only categorical cols.
    """
    base = preprocess_raw(df_full[REQUIRED_COLS])
    cat_encoded = pd.get_dummies(base[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, drop_first=True)
    numeric_part = base.drop(columns=CATEGORICAL_COLS)
    full = pd.concat([numeric_part, cat_encoded], axis=1)
    return full.columns.tolist()

def get_expected_feature_names(model) -> list[str] | None:
    base = extract_underlying_estimator(model)
    names = getattr(base, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(base, Pipeline):
        last = base[-1]
        names = getattr(last, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None

def to_dummies_aligned(df_pre: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    cat_encoded = pd.get_dummies(df_pre[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, drop_first=True)
    Xd = pd.concat([df_pre.drop(columns=CATEGORICAL_COLS), cat_encoded], axis=1)
    return Xd.reindex(columns=expected_cols, fill_value=0)

def prepare_X(df_raw: pd.DataFrame,
              model,
              fallback_dummy_cols: list[str]) -> pd.DataFrame:
    """
    Preprocess -> manual one-hot on holiday/weather_main -> align to model features.
    """
    base = preprocess_raw(df_raw[REQUIRED_COLS])
    expected = get_expected_feature_names(model)
    if expected is not None and len(expected) > 0:
        expected_set = set(expected)
        # If model was trained on raw (unlikely), pass those as-is
        if expected_set == set(base.columns) and len(expected) == len(base.columns):
            return base
        return to_dummies_aligned(base, expected)
    # Fall back to a template built from training data
    return to_dummies_aligned(base, fallback_dummy_cols)

def mapie_conformalize_prefit(reg: BaseEstimator,
                              X_cf: pd.DataFrame,
                              y_cf: pd.Series) -> MapieRegressor:
    """
    Wraps a prefit estimator with MAPIE (split/prefit style).
    """
    mapie = MapieRegressor(estimator=reg, cv="prefit", n_jobs=-1)
    mapie.fit(X_cf, y_cf)
    return mapie

def predict_with_intervals(mapie_model: MapieRegressor, X: pd.DataFrame, alpha: float):
    """
    Returns y_pred (1D) and y_int (n,2) using MAPIE's modern API.
    """
    y_pred, y_pis = mapie_model.predict(X, alpha=alpha)
    # y_pis shape: (n, 2, n_alpha); here n_alpha==1 -> take [:,:,0]
    if y_pis.ndim == 3 and y_pis.shape[1] == 2:
        y_int = y_pis[:, :, 0]
    elif y_pis.ndim == 2 and y_pis.shape[1] == 2:
        y_int = y_pis
    else:
        raise ValueError(f"Unexpected MAPIE interval shape: {y_pis.shape}")
    return y_pred, y_int

# ---- Sidebar: Settings + Inputs ----
with st.sidebar:
    st.header("⚙️ Settings")
    alpha = st.slider(
        "Significance level α (intervals widen as α ↓)",
        min_value=0.01, max_value=0.20, value=0.10, step=0.01,
        help="Prediction interval coverage ≈ 1 − α (e.g., α=0.10 ⇒ ~90% coverage)."
    )
    st.caption("Model: XGBoost | Intervals: MAPIE (prefit)")

    # OPTIONAL: add a sidebar image if you have it
    # st.image("traffic_sidebar.jpg", use_column_width=True)

    st.markdown("---")

    # Manual Input (moved to sidebar)
    with st.expander("📝 Manual Input (Form)", expanded=True):
        st.caption("Provide weather & holiday conditions. Date/Time drives rush-hour patterns.")
        holiday = st.selectbox(
            "Holiday",
            options=["None", "New Years Day", "Martin Luther King Jr Day", "Memorial Day",
                     "Independence Day", "Labor Day", "Columbus Day", "Veterans Day",
                     "Thanksgiving Day", "Christmas Day"],
            index=0
        )
        temp = st.number_input("Temperature (Kelvin)", min_value=200.0, max_value=330.0, value=288.0, step=0.5)
        rain_1h = st.number_input("Rain in last hour (mm)", min_value=0.0, value=0.0, step=0.1)
        snow_1h = st.number_input("Snow in last hour (mm)", min_value=0.0, value=0.0, step=0.1)
        clouds_all = st.number_input("Cloud coverage (0-100)", min_value=0, max_value=100, value=40, step=1)
        weather_main = st.selectbox(
            "Weather",
            options=["Clear", "Clouds", "Rain", "Snow", "Drizzle", "Mist", "Thunderstorm", "Haze", "Fog"],
            index=1
        )

        colD, colT = st.columns(2)
        with colD:
            d_in = st.date_input("Date", value=pd.Timestamp("2012-10-02").date())
        with colT:
            t_in = st.time_input("Time", value=pd.Timestamp("2012-10-02 09:00").time())

        manual_submit = st.button("Predict (Manual Input)")

    st.markdown("---")

    # CSV Upload (moved to sidebar)
    with st.expander("📂 CSV Upload", expanded=True):
        st.caption("Your CSV must contain these columns:")
        st.code(", ".join(REQUIRED_COLS), language="text")

        sample = pd.DataFrame([{
            "holiday": "None",
            "temp": 288.28,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "date_time": "10/2/12 9:00"
        }])
        st.dataframe(sample)

        uploaded = st.file_uploader("Upload a CSV with the columns above", type=["csv"])
        csv_submit = st.button("Predict (CSV Upload)")

# ---- Load model & data ----
base_model = None
traffic_df = None

try:
    base_model = load_base_model(PICKLE_PATH)
except Exception as e:
    st.error(f"Could not load model from `{PICKLE_PATH}`.\n\n{e}")

try:
    traffic_df = load_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not load `{DATA_PATH}`.\n\n{e}")

if base_model is None or traffic_df is None:
    st.stop()

ok_cols, missing = ensure_required_columns(traffic_df, need_target=True)
if not ok_cols:
    st.warning(f"`{DATA_PATH}` is missing required columns: {missing}.\n"
               f"Expected columns: {ALL_COLS_WITH_TARGET}")
    st.stop()

# Build template dummy column list from the training data (preprocessed)
dummy_cols = build_dummy_template(traffic_df)

# Internal split for diagnostics or (re)fitting if needed
df_train, df_test = train_test_split(traffic_df, test_size=0.2, random_state=42)
y_all  = df_train["traffic_volume"]
y_test = df_test["traffic_volume"]

X_all  = prepare_X(df_train, base_model, fallback_dummy_cols=dummy_cols)
X_test = prepare_X(df_test,  base_model, fallback_dummy_cols=dummy_cols)

# If the loaded model is not prefit, fit it now on part of X_all
try:
    _ = base_model.predict(X_all.head(1))
    prefit_ok = True
except Exception:
    prefit_ok = False

trained = base_model
X_tr, X_cf, y_tr, y_cf = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

if not prefit_ok:
    try:
        trained.fit(X_tr, y_tr)
        st.info(f"Model in `{PICKLE_PATH}` did not appear pre-fitted; trained on {len(X_tr)} samples.")
        prefit_ok = True
    except Exception as e:
        st.error(f"Model cannot predict and failed to fit on provided data.\n\n{e}")
        st.stop()

# Conformalize (prefit)
try:
    conformal_model = mapie_conformalize_prefit(trained, X_cf, y_cf)
except Exception as e:
    st.error(f"Failed to conformalize model.\n\n{e}")
    st.stop()

trained_model_for_importance = trained


# ---- Manual Prediction ----
if manual_submit:
    # Build manual row in the dataset’s schema
    dt = pd.to_datetime(f"{d_in} {t_in}")  # robust on all platforms
    # Format like the dataset "10/2/12 9:00" (no leading zeros)
    date_time_str = f"{dt.month}/{dt.day}/{dt.strftime('%y')} {dt.hour}:{dt.strftime('%M')}"

    manual_raw = pd.DataFrame([{
        "holiday": holiday,
        "temp": float(temp),
        "rain_1h": float(rain_1h),
        "snow_1h": float(snow_1h),
        "clouds_all": int(clouds_all),
        "weather_main": weather_main,
        "date_time": date_time_str
    }])

    try:
        manual_X = prepare_X(manual_raw, trained, fallback_dummy_cols=dummy_cols)
        y_pred, y_int = predict_with_intervals(conformal_model, manual_X, alpha=alpha)
        st.success(f"Predicted Volume: {float(y_pred[0]):,.0f}")
        st.info(f"{int(100*(1-alpha))}% Prediction Interval (α={alpha:.2f}): "
                f"{float(y_int[0,0]):,.0f} — {float(y_int[0,1]):,.0f}")
    except Exception as e:
        st.error(f"Manual prediction failed: {e}")


# ---- CSV Prediction ----
if csv_submit and uploaded is not None:
    try:
        user_df_raw = pd.read_csv(uploaded)
        ok, missing = ensure_required_columns(user_df_raw, need_target=False)
        if not ok:
            st.error(f"Uploaded CSV missing columns: {missing}")
        else:
            user_X = prepare_X(user_df_raw, trained, fallback_dummy_cols=dummy_cols)
            y_pred, y_int = predict_with_intervals(conformal_model, user_X, alpha=alpha)
            out = user_df_raw.copy()
            out["Predicted Volume"] = np.round(y_pred, 2)
            out["Lower Prediction Limit"] = np.round(y_int[:, 0], 2)
            out["Upper Prediction Limit"] = np.round(y_int[:, 1], 2)
            st.success("CSV predictions complete. Showing first 20 rows:")
            st.dataframe(out.head(20))
    except Exception as e:
        st.error(f"Failed to process uploaded CSV: {e}")

st.markdown("---")

# ---- Diagnostics & Insights ----
st.header("Model Diagnostics & Insights")

# 1) Feature Importance
st.subheader("Feature Importance (XGBoost)")
importances, feat_names = safe_feature_importances(trained_model_for_importance, X_all.columns.tolist())
if importances.size > 0:
    order = np.argsort(importances)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.barh(np.array(feat_names)[order], importances[order])
    ax1.set_xlabel("Importance")
    ax1.set_title("XGBoost Feature Importances")
    st.pyplot(fig1)
else:
    st.info("Feature importances not available for this model.")

# 2) Performance on internal test split
st.subheader("Performance on Internal Test Split")
try:
    y_test_pred, y_test_int = predict_with_intervals(conformal_model, X_test, alpha=alpha)

    pred_df = pd.DataFrame({
        "Actual Value":  np.asarray(y_test).astype(float),
        "Predicted":     np.asarray(y_test_pred).astype(float),
        "Lower Value":   np.asarray(y_test_int[:, 0]).astype(float),
        "Upper Value":   np.asarray(y_test_int[:, 1]).astype(float),
    })
    sorted_pred = pred_df.sort_values(by="Actual Value").reset_index(drop=True)

    residuals = sorted_pred["Actual Value"] - sorted_pred["Predicted"]
    colA, colB = st.columns(2)
    with colA:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(residuals, bins=40)
        ax2.set_title("Residuals Histogram")
        ax2.set_xlabel("Error (y - ŷ)")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    with colB:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(sorted_pred["Actual Value"], sorted_pred["Predicted"], s=10, alpha=0.6)
        mn = float(sorted_pred[["Actual Value", "Predicted"]].min().min())
        mx = float(sorted_pred[["Actual Value", "Predicted"]].max().max())
        ax3.plot([mn, mx], [mn, mx])
        ax3.set_title("Predicted vs Actual")
        ax3.set_xlabel("Actual Volume")
        ax3.set_ylabel("Predicted Volume")
        st.pyplot(fig3)

    lower = y_test_int[:, 0]
    upper = y_test_int[:, 1]
    cov = regression_coverage_score(np.asarray(y_test).astype(float), lower, upper)
    coverage_percentage = 100.0 * float(np.asarray(cov).squeeze())

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x_idx = np.arange(len(sorted_pred))
    ax4.fill_between(x_idx, sorted_pred["Lower Value"], sorted_pred["Upper Value"],
                     alpha=0.2, label="Prediction Interval")
    ax4.plot(sorted_pred["Actual Value"], "go", markersize=3, label="Actual Value")
    ax4.set_xlim([0, len(sorted_pred)])
    ax4.set_xlabel("Samples", fontsize=10)
    ax4.set_ylabel("Traffic Volume", fontsize=10)
    ax4.set_title(f"Coverage Plot: {coverage_percentage:.2f}% (Target ≈ {(1-alpha)*100:.0f}%)",
                  fontsize=12, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=10)
    st.pyplot(fig4)

except Exception as e:
    st.info(f"Could not produce diagnostic plots: {e}")
