==========================================================
OPEN-DATA SIGNALS – INDIAN EQUITY TIMING
==========================================================

Author: Shashwat Gautam(221005)
Date: [10/11/2025]
Environment: Python 3.11+
----------------------------------------------------------

PROJECT OVERVIEW
----------------------------------------------------------
This project implements an end-to-end machine learning pipeline
for weekly Indian equity timing. It predicts the next week’s
returns for four major indices:
    • Momentum
    • Quality
    • Value
    • SmallCap

The system uses mutual fund flow data and price data to generate
signals, trains ML models (ElasticNet and LightGBM), simulates
a weekly trading strategy through backtesting, and visualizes
performance using an interactive Streamlit web dashboard.

==========================================================
CLONE THE GIT REPO
==========================================================
git clone https://github.com/ShashwatG04/open-data-signal
cd open-data-signals

==========================================================
ENVIRONMENT SETUP
==========================================================
1. Create and activate virtual environment:
    python -m venv venv311
    venv311\Scripts\activate

2. Install all dependencies:
    pip install -r requirements.txt

==========================================================
Now you have three options:
==========================================================

==========================================================
2️⃣ DOUBLE CLICK THE RUN.BAT FILE
==========================================================
Navifate to the file explorer and just double click the run.bat file.

==========================================================
2️⃣ RUN THE RUN.PY FILE
==========================================================
Execute run.py file using python(while the venv311) is active.
This file includes the whole pipeline execution in correct order.

==========================================================
3️⃣ RUNNING THE FULL PIPELINE
==========================================================

STEP 1: Build Weekly Dataset
----------------------------
Command:
    python data_pipeline.py

Purpose:
    • Loads or creates synthetic AMFI flow data and index price data.
    • Calculates weekly price returns.
    • Saves a merged weekly dataset to:
        → ./data/weekly_panel.parquet

Output Columns:
    Date, price_<Index>, flow_pressure_<Index>, return_1w_<Index>


STEP 2: Feature Engineering
----------------------------
Command:
    python features.py

Purpose:
    • Converts the weekly panel into long format.
    • Creates predictive features:
        r_1w   → 1-week return
        r_4w   → 4-week average return
        r_12w  → 12-week average return
        vol_4w → 4-week volatility
        flow_z → 12-week z-score of fund flow
    • Creates 'target' = next week's return.

Output:
    ./data/features.parquet


STEP 3: Model Training
----------------------------
Command:
    python train.py

Purpose:
    • Trains two regression models:
        1. ElasticNet  (linear model with L1/L2 regularization)
        2. LightGBM    (non-linear gradient boosting model)
    • Uses engineered features:
        [r_1w, r_4w, r_12w, flow_z, vol_4w]
    • Saves trained models to:
        ./models/elasticnet.pkl
        ./models/lgbm.pkl


STEP 4: Backtesting
----------------------------
Command:
    python backtest.py

Purpose:
    • Loads the trained model (LightGBM by default).
    • Predicts next-week returns for each index.
    • Smooths predictions using an 8-period exponential moving average (EMA).
    • Generates long/short positions using signal thresholds.
    • Applies transaction cost (0.1% default).
    • Computes cumulative Portfolio Value and Buy & Hold benchmark.

Output:
    ./data/backtest_results.csv

Key Columns:
    Date, index, predicted_return, position, strategy_return, Portfolio_Value, buy_hold


STEP 5: Visualization Dashboard
----------------------------
Command:
    streamlit run app.py
OR
    Double-click 'run.bat'

Purpose:
    • Interactive visualization built with Streamlit.
    • Shows:
        - Sample of engineered features.
        - Model predictions (1-week-ahead returns).
        - Correlation between predictions and true returns.
        - Backtest performance chart (Portfolio Value vs. Date).
    • Sidebar allows model selection (ElasticNet or LightGBM).


==========================================================
==========================================================
FILE-BY-FILE EXPLANATION
==========================================================
==========================================================

data_pipeline.py
----------------
Builds the initial dataset.
Fetches or simulates index prices and mutual fund flow data.
Computes weekly returns and creates a “weekly panel”.
Output → weekly_panel.parquet

features.py
------------
Processes the panel data into features used for modeling.
Adds rolling averages, volatility, and flow z-scores.
Defines the “target” (next week’s return).
Output → features.parquet

train.py
---------
Loads features, trains models using scikit-learn and LightGBM.
ElasticNet → Linear baseline.
LightGBM → Gradient boosting trees (handles non-linearity).
Outputs trained models as .pkl files.

backtest.py
------------
Simulates trading using model predictions.
Applies position logic:
    - Go LONG if smoothed signal > +threshold
    - Go SHORT if smoothed signal < -threshold
    - Stay NEUTRAL otherwise
Computes returns, applies transaction cost, and tracks
Portfolio Value over time.
Outputs → backtest_results.csv

utils.py
---------
Contains utility functions for:
    - Loading / saving CSV or Parquet files
    - Ensuring directory structure
    - Seeding for reproducibility

app.py
------
Streamlit dashboard that ties everything together.
Displays feature samples, predictions, correlation, and
cumulative backtest performance in an interactive layout.

run.bat
-------
Simple batch file to activate the virtual environment and
launch the Streamlit app automatically.

requirements.txt
----------------
Lists all dependencies required to run the project.


==========================================================
==========================================================
HOW IT WORKS INTERNALLY
==========================================================
==========================================================

1. Data from AMFI and market indices are merged weekly.
2. Feature engineering creates rolling averages and volatility.
3. Machine learning models learn to predict next week’s returns.
4. Predictions are smoothed using exponential moving average (EMA).
5. Positions are taken based on signal thresholds.
6. Weekly portfolio value is updated and tracked cumulatively.
7. Streamlit app visualizes model performance.

==========================================================
==========================================================
TECHNICAL DETAILS
==========================================================
==========================================================

Core Libraries:
    - pandas: Data manipulation
    - numpy: Numerical computation
    - scikit-learn: ElasticNet regression
    - lightgbm: Gradient boosting model
    - joblib: Model saving/loading
    - streamlit: Web visualization

Key ML Techniques:
    • ElasticNet regression for baseline linear modeling.
    • LightGBM (tree-based gradient boosting) for non-linear signal patterns.
    • Exponential Moving Average smoothing on predictions.
    • Rolling mean and volatility for temporal feature creation.
    • Z-score normalization for flow signals.
    • Backtesting with transaction cost simulation.

Expected Time Series Frequency:
    Weekly (monthly data is also supported).

Main Outputs:
    data/features.parquet
    data/backtest_results.csv
    models/elasticnet.pkl
    models/lgbm.pkl

==========================================================
==========================================================
COMPLETE COMMAND SEQUENCE
==========================================================
==========================================================

python data_pipeline.py
python features.py
python train.py
python backtest.py
streamlit run app.py


==========================================================
==========================================================
FINAL NOTES
==========================================================
==========================================================

• All code modules are modular and interchangeable.
• You can substitute real AMFI and index data for synthetic data.
• The model is designed to predict one-week-ahead returns.
• The pipeline can be extended to daily or monthly frequency.
• The Streamlit app can be deployed easily to Streamlit Cloud.
==========================================================


