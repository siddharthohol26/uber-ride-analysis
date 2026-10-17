import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Uber Ride Analytics", page_icon="🚕", layout="wide")
st.title("🚕 Uber Ride Predictive Analytics Dashboard")
st.caption("Mini project: data cleaning • exploratory analysis • correlation • prediction")

st.sidebar.header("1. Upload dataset")
uploaded = st.sidebar.file_uploader("Upload a Uber CSV file", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(name in cl for name in candidates):
            return c
    return None

if uploaded is None:
    st.info("Upload your Uber Rides CSV from Kaggle to begin.")
    st.markdown("""
    ### What this mini project does
    - Cleans missing values and duplicate rows
    - Shows ride volume, time patterns, and pickup locations when available
    - Displays numeric correlations
    - Trains a regression model when a numeric target such as `trips` is available
    - Trains a simple peak-hour classifier when date/time information is available

    **Common supported datasets**
    - Uber trip-count data with columns such as `date`, `trips`, `active_vehicles`
    - Uber pickup data with columns such as `Date/Time`, `Lat`, `Lon`, `Base`

    **Tip:** Keep the original CSV column names; the dashboard detects common variants automatically.
    """)
    st.stop()

try:
    raw = load_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if raw.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

df = raw.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()

date_col = find_col(df, ["date/time", "datetime", "timestamp", "date", "pickup_datetime", "pickup_date"])
trips_col = find_col(df, ["trips", "trip_count", "rides", "ride_count", "total_trips"])
vehicles_col = find_col(df, ["active_vehicles", "vehicles"])
lat_col = find_col(df, ["lat", "latitude"])
lon_col = find_col(df, ["lon", "lng", "longitude"])
base_col = find_col(df, ["base", "dispatching_base_number"])

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].notna().any():
        df["__hour"] = df[date_col].dt.hour
        df["__day"] = df[date_col].dt.day
        df["__weekday"] = df[date_col].dt.day_name()
        df["__month"] = df[date_col].dt.month
        df["__date_only"] = df[date_col].dt.date

st.sidebar.header("2. Dashboard filters")
if date_col and df[date_col].notna().any():
    valid_dates = df[date_col].dropna()
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    selected = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(selected, tuple) and len(selected) == 2:
        start, end = selected
        df = df[df[date_col].dt.date.between(start, end)]

st.subheader("Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows after cleaning", f"{len(df):,}")
b.metric("Columns", len(df.columns))
c.metric("Duplicate rows removed", f"{len(raw) - len(raw.drop_duplicates()):,}")
d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

with st.expander("Preview data and data quality"):
    st.dataframe(df.head(100), use_container_width=True)
    quality = pd.DataFrame({"Missing values": df.isna().sum(), "Missing %": (df.isna().mean()*100).round(2), "Data type": df.dtypes.astype(str)})
    st.dataframe(quality, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Explore", "🔗 Correlation", "🤖 Predict", "📍 Pickups"])

with tab1:
    st.subheader("Exploratory data analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if trips_col and trips_col in df.columns:
        y = pd.to_numeric(df[trips_col], errors="coerce")
        st.write(f"**Trip count summary — {trips_col}**")
        st.dataframe(y.describe().to_frame("value"), use_container_width=True)
        fig, ax = plt.subplots()
        y.dropna().plot(kind="hist", bins=30, ax=ax)
        ax.set_xlabel(trips_col)
        ax.set_title("Distribution of trip counts")
        st.pyplot(fig)
        plt.close(fig)
    if date_col and df[date_col].notna().any():
        st.write("**Ride activity over time**")
        if trips_col and trips_col in df.columns:
            series = df.dropna(subset=[date_col]).copy()
            series[trips_col] = pd.to_numeric(series[trips_col], errors="coerce")
            series = series.dropna(subset=[trips_col]).sort_values(date_col)
            if len(series):
                fig, ax = plt.subplots()
                ax.plot(series[date_col], series[trips_col])
                ax.set_xlabel("Date/time")
                ax.set_ylabel("Trips")
                ax.set_title("Trips over time")
                fig.autofmt_xdate()
                st.pyplot(fig)
                plt.close(fig)
        if "__hour" in df.columns:
            hourly = df.groupby("__hour").size().reindex(range(24), fill_value=0)
            fig, ax = plt.subplots()
            hourly.plot(kind="bar", ax=ax)
            ax.set_title("Records by hour of day")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Records")
            st.pyplot(fig)
            plt.close(fig)
        if "__weekday" in df.columns:
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            counts = df["__weekday"].value_counts().reindex(weekday_order).dropna()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_title("Records by day of week")
            ax.set_ylabel("Records")
            st.pyplot(fig)
            plt.close(fig)
    if vehicles_col and trips_col and vehicles_col in df.columns and trips_col in df.columns:
        temp = df[[vehicles_col, trips_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(temp):
            fig, ax = plt.subplots()
            ax.scatter(temp[vehicles_col], temp[trips_col], alpha=0.6)
            ax.set_xlabel(vehicles_col)
            ax.set_ylabel(trips_col)
            ax.set_title("Active vehicles vs trips")
            st.pyplot(fig)
            plt.close(fig)
    if not numeric_cols:
        st.warning("No numeric columns were detected. Check that the CSV has been parsed correctly.")

with tab2:
    st.subheader("Correlation mapping")
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        st.warning("At least two numeric columns are required for correlation analysis.")
    else:
        corr = numeric.corr(numeric_only=True)
        st.dataframe(corr.round(3), use_container_width=True)
        fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)*0.7), max(4, len(corr.columns)*0.55)))
        im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        ax.set_title("Numeric feature correlation heatmap")
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Correlation shows association, not proof that one variable causes another.")

with tab3:
    st.subheader("Predictive modelling")
    st.markdown("The model below uses available numeric features. Results depend on the quality and size of the uploaded dataset.")
    numeric = df.select_dtypes(include=np.number).copy()
    # exclude generated time fields only if they are not meaningful; hour/day/month can help prediction
    target_options = [c for c in numeric.columns if numeric[c].nunique(dropna=True) > 1]
    default_target = trips_col if trips_col in target_options else (target_options[0] if target_options else None)
    if default_target is None:
        st.warning("No suitable numeric target found. Use a dataset with a numeric field such as `trips`.")
    else:
        target = st.selectbox("Choose numeric target to predict", target_options, index=target_options.index(default_target))
        feature_options = [c for c in numeric.columns if c != target]
        if date_col and df[date_col].notna().any():
            for c in ["__hour", "__day", "__month"]:
                if c in df.columns and c not in feature_options:
                    numeric[c] = df[c]
                    feature_options.append(c)
        if not feature_options:
            st.warning("At least one numeric feature besides the target is required.")
        else:
            chosen_features = st.multiselect("Choose features", feature_options, default=feature_options[:min(5, len(feature_options))])
            if st.button("Train regression model", type="primary"):
                if not chosen_features:
                    st.error("Choose at least one feature.")
                else:
                    model_data = numeric[chosen_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(model_data) < 10:
                        st.error("At least 10 complete rows are recommended to train the model.")
                    else:
                        X, y = model_data[chosen_features], model_data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = RandomForestRegressor(n_estimators=120, random_state=42)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
                        m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}")
                        m3.metric("R² score", f"{r2_score(y_test, pred):.3f}")
                        results = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
                        st.dataframe(results.head(30), use_container_width=True)
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, pred, alpha=0.7)
                        ax.set_xlabel("Actual values"); ax.set_ylabel("Predicted values")
                        ax.set_title("Actual vs predicted")
                        st.pyplot(fig); plt.close(fig)
                        importance = pd.Series(model.feature_importances_, index=chosen_features).sort_values()
                        fig, ax = plt.subplots()
                        importance.plot(kind="barh", ax=ax)
                        ax.set_title("Feature importance")
                        st.pyplot(fig); plt.close(fig)
                        st.success("Model trained. For formal forecasting, use a time-based train/test split to avoid future-data leakage.")

    st.divider()
    st.markdown("**Optional peak-hour classification**")
    if "__hour" in df.columns and df["__hour"].notna().any():
        st.write("A simple peak label is created from pickup hour. This is a demonstration label, not a label supplied by Uber.")
        threshold = st.slider("Peak hour threshold (records per hour)", min_value=1, max_value=100, value=10)
        hour_counts = df.groupby("__hour").size()
        peak_hours = hour_counts[hour_counts >= threshold].index.tolist()
        st.write("Hours meeting the threshold:", peak_hours if peak_hours else "None")
    else:
        st.caption("Classification example requires a parseable date/time column.")

with tab4:
    st.subheader("Pickup location analysis")
    if lat_col and lon_col:
        points = df[[lat_col, lon_col]].copy()
        points[lat_col] = pd.to_numeric(points[lat_col], errors="coerce")
        points[lon_col] = pd.to_numeric(points[lon_col], errors="coerce")
        points = points.dropna()
        points = points[points[lat_col].between(-90, 90) & points[lon_col].between(-180, 180)]
        st.metric("Valid coordinate pairs", f"{len(points):,}")
        if len(points):
            st.map(points.rename(columns={lat_col: "lat", lon_col: "lon"}).head(5000), use_container_width=True)
            st.caption("Map displays up to 5,000 points to keep the dashboard responsive.")
    else:
        st.info("No latitude/longitude columns detected. Use a pickup dataset containing `Lat` and `Lon` to display a map.")
    if base_col:
        st.write("Records by base")
        st.bar_chart(df[base_col].value_counts().head(15))

st.sidebar.markdown("---")
st.sidebar.caption("Mini project • Python + Pandas + Scikit-learn + Streamlit")
st.sidebar.caption(" © 2026 Siddharth Ohol | All Rights Reserved")
