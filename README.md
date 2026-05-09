# 5291 Final Project: Financial Time Series Prediction Using Machine Learning and Numerical Validation Methods

## Project Overview

This project studies financial time series prediction using multiple asset classes, including precious metals, energy, a currency index, and cryptocurrencies. The full workflow includes data cleaning, preprocessing, exploratory data analysis, feature engineering, unsupervised learning, supervised machine learning, numerical method validation, and an interactive Shiny web application.

The six assets analyzed in this project are:

| Asset | Yahoo Finance Ticker | Category |
|---|---|---|
| Gold | GC=F | Precious Metal |
| Silver | SI=F | Precious Metal |
| WTI Oil | CL=F | Energy |
| USD Index | DX-Y.NYB | Currency Index |
| Bitcoin | BTC-USD | Cryptocurrency |
| Ethereum | ETH-USD | Cryptocurrency |

The main prediction target is the next-day log return of each asset. Predicted returns are also converted into predicted price paths for interpretation and numerical validation.

---

## Live Web Application

The interactive Shiny web application is available here:

[https://baixuanchen5243.shinyapps.io/5291-Final-Project/](https://baixuanchen5243.shinyapps.io/5291-Final-Project/)

The app allows users to explore:

- Data cleaning and preprocessing outputs
- Exploratory data analysis figures
- Feature engineering results
- Unsupervised learning outputs
- Supervised model comparison tables
- Return prediction plots
- Price prediction plots
- Numerical validation results

---

## Repository Structure

```text
5291_final-project/
├── app.R
├── README.md
├── 5291 Final Project code.ipynb
├── web/
├── result/
│   ├── clean_process_result/
│   ├── eda_figures/
│   ├── feature_engineer_result/
│   ├── modeling_outputs/
│   ├── numercial_output/
│   └── unsupervised_analysis_result/
└── report/
    ├── main.tex
    ├── EDA_plot/
    ├── feature_engineering_plot/
    ├── modeling_plot/
    ├── modeling_return_plot/
    ├── numerical_validation_plot/
    └── unsupervised_analysis_plot/
```

---

## Main Files and Folders

### `app.R`

This is the main R Shiny application file. It loads project figures and result tables and presents the project workflow in an interactive web interface.

### `5291 Final Project code.ipynb`

This notebook contains the main Python analysis workflow, including data cleaning, EDA, feature engineering, unsupervised learning, supervised modeling, and numerical validation.

### `web/`

This folder contains the files used by the Shiny web application, including cleaned data, result tables, and figures required for the deployed app.

### `result/`

This folder stores the main outputs generated from the analysis pipeline.

- `clean_process_result/`: cleaned price data, log returns, train-test split outputs, and scaled feature matrices
- `eda_figures/`: exploratory data analysis plots
- `feature_engineer_result/`: extended feature engineering outputs and scaled extended features
- `modeling_outputs/`: supervised model predictions, model comparison tables, and modeling figures
- `numercial_output/`: numerical method validation results and validation plots
- `unsupervised_analysis_result/`: PCA, clustering, and outlier detection outputs

> Note: The folder name is written as `numercial_output` to match the current repository folder name.

### `report/`

This folder contains the LaTeX source code and figures used for the final project report.

---

## Methods

### 1. Data Cleaning and Preprocessing

Daily price data were collected from Yahoo Finance using Python. The assets were aligned by date, missing values were handled, and prices were transformed into daily log returns:

```text
log_return_t = log(price_t) - log(price_{t-1})
```

A chronological train-test split was used because this is a time series prediction problem. This avoids using future information during model training.

---

### 2. Exploratory Data Analysis

The exploratory data analysis examines:

- Price level time series
- Normalized price series
- Daily log returns
- Rolling volatility
- Return distributions
- Correlation heatmap
- Risk-return relationship
- Rolling correlation with Bitcoin

The EDA results show that cryptocurrencies have higher volatility and more extreme return behavior, while Gold and USD Index are relatively more stable.

---

### 3. Feature Engineering

The feature engineering stage creates a larger market feature set, including:

- Lagged returns
- Rolling mean and rolling volatility
- Momentum features
- Volatility ratio features
- Rolling correlations
- Drawdown features
- Extreme-return indicators
- Market regime indicators

These features are used to capture temporal dependence, changing volatility, cross-asset relationships, and unusual market conditions.

---

### 4. Unsupervised Learning

The unsupervised analysis uses:

- Principal Component Analysis
- K-Means Clustering
- Isolation Forest Outlier Detection

PCA is used to reduce the high-dimensional feature space. K-Means clustering identifies market condition groups, while Isolation Forest detects unusual observations.

---

### 5. Supervised Machine Learning

Three supervised learning models are compared:

| Model | Description |
|---|---|
| Ridge Regression | Linear benchmark with L2 regularization |
| Random Forest | Nonlinear tree-based ensemble model |
| Hist Gradient Boosting | Boosted tree-based model |

The models predict next-day log returns for each asset. They are evaluated using:

- MSE
- RMSE
- MAE
- R-squared
- Directional accuracy

Overall, Random Forest achieves the best RMSE for most assets, while Hist Gradient Boosting performs best for Silver.

---

### 6. Numerical Method Validation

Numerical validation converts predicted returns into predicted price paths. Three updating methods are compared:

| Method | Description |
|---|---|
| Euler | Direct one-step return update |
| Trapezoidal | Average-based return update |
| RK4-like | Weighted update based on intermediate approximations |

The Euler method is the most reliable overall because the supervised models predict discrete daily returns rather than continuous-time derivatives.

---

## Main Results

The best supervised model for each asset is:

| Asset | Best Model |
|---|---|
| Gold | Random Forest |
| Silver | Hist Gradient Boosting |
| WTI Oil | Random Forest |
| USD Index | Random Forest |
| Bitcoin | Random Forest |
| Ethereum | Random Forest |

The results suggest that nonlinear tree-based models are more effective than the linear Ridge Regression benchmark for the engineered financial feature set.

---

## How to Run the Shiny App Locally

### 1. Clone the repository

```bash
git clone https://github.com/Trigger-chen/5291_final-project.git
cd 5291_final-project
```

### 2. Install required R packages

Run the following code in R or RStudio:

```r
install.packages(c(
  "shiny",
  "shinydashboard",
  "shinyjs",
  "bslib",
  "DT",
  "plotly",
  "ggplot2",
  "dplyr",
  "tidyr",
  "readr",
  "readxl",
  "jsonlite",
  "tools"
))
```

### 3. Run the app

Open `app.R` in RStudio and click **Run App**.

Alternatively, run:

```r
shiny::runApp("app.R")
```

---

## How to Reproduce the Python Analysis

The analysis was developed using Python. Main packages include:

- pandas
- numpy
- matplotlib
- yfinance
- scikit-learn
- statsmodels
- arch
- openpyxl

Install the required packages using:

```bash
pip install pandas numpy matplotlib yfinance scikit-learn statsmodels arch openpyxl
```

Then run:

```text
5291 Final Project code.ipynb
```

The notebook generates the cleaned datasets, figures, model predictions, result tables, and numerical validation outputs.

---

## Final Report

The final report source files are stored in:

```text
report/
```

Main LaTeX file:

```text
report/main.tex
```

The `report/` folder also contains all figures used in the final PDF report.

---

## Authors

| Name | UNI |
|---|---|
| Baixuan Chen | bc3212 |
| Purui Niu | pn2433 |
| Junyang Li | jl7230 |

---

## Conclusion

This project shows that combining feature engineering, nonlinear supervised learning, unsupervised market structure analysis, numerical validation, and an interactive Shiny application provides a useful framework for financial time series prediction. Although short-term financial returns remain difficult to forecast, Random Forest with engineered financial features and Euler-based price reconstruction provided the most stable overall performance in this project.