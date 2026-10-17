# Uber Ride Predictive Analytics Dashboard — Mini Project

## Features
- Upload a Kaggle Uber rides CSV
- Remove duplicate records and inspect missing values
- Exploratory data analysis (EDA)
- Time-pattern charts for datasets containing date/time
- Numeric correlation matrix and heatmap
- Random Forest regression with MAE, RMSE, R², actual-vs-predicted chart, and feature importance
- Pickup map for datasets containing latitude and longitude
- Optional peak-hour exploration

## Requirements
- Python 3.10 or newer recommended
- Internet only for downloading Python packages / dataset; dashboard itself runs locally

## Run on macOS
1. Extract the ZIP file.
2. Open Terminal and move into the project folder:
   ```bash
   cd uber_predictive_dashboard
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
6. In the browser, upload the Uber CSV downloaded from Kaggle.

## Dataset
Use the Uber Rides Dataset supplied by your course/Kaggle. The dashboard detects common column names automatically, but the exact charts and prediction targets depend on the CSV schema.

Common supported formats:
- Trip totals: `date`, `trips`, `active_vehicles`
- Pickup points: `Date/Time`, `Lat`, `Lon`, `Base`

## Suggested report outline
1. Title and objective
2. Dataset description
3. Tools and technologies
4. Data cleaning
5. Exploratory analysis
6. Correlation mapping
7. Regression model and evaluation
8. Results and limitations
9. Conclusion

## Important notes
- Model metrics are generated from your uploaded data; no accuracy claims are pre-filled.
- Correlation is association, not causation.
- If the dataset only has pickup timestamps and coordinates, it does not contain a true fare/trip-count target. The regression section can predict an available numeric field, but that may not be meaningful. For trip-demand forecasting, aggregate rides by hour/day first and use those counts as the target.
