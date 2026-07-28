# 🏠 Boston Housing Price Prediction

An end-to-end **regression** project that predicts median house prices (`MEDV`) using the classic Boston Housing dataset. The notebook covers data loading, EDA, feature transformation, and comparison of multiple regression models with hyperparameter tuning.

---

## 📌 Problem Statement

Given housing and neighborhood attributes (crime rate, rooms, pollution, tax rate, etc.), predict the continuous target variable **`MEDV`** — median value of owner-occupied homes (in $1000s).

---

## 📊 About the Dataset

The dataset is the well-known **Boston Housing dataset** (`housing.csv`), loaded with whitespace-delimited columns and manually assigned headers.

| Column | Description |
|---|---|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots *(dropped)* |
| `INDUS` | Proportion of non-retail business acres *(dropped)* |
| `CHAS` | Bounds Charles River (1 = Yes, 0 = No) *(dropped)* |
| `NOX` | Nitric oxide concentration (pollution level) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 *(dropped)* |
| `DIS` | Weighted distance to employment centers |
| `RAD` | Index of accessibility to radial highways *(dropped)* |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Demographic factor (proportion of Black residents by town) |
| `LSTAT` | % of lower-status population |
| `MEDV` | **Target** — median home value ($1000s) |

**Key relationships (from EDA):**
- `RM` (more rooms) → higher price 📈
- `LSTAT` (more lower-income population) → lower price 📉
- `PTRATIO` → reflects education quality impact on price

**Data characteristics:** mostly numerical, no major missing values, some features skewed, contains outliers.

---

## 🔄 Project Workflow

### 1. Data Loading
- Libraries: `pandas`, `numpy`
- Loaded `housing.csv` with manually defined column names (whitespace-delimited format)

### 2. Initial Inspection
- `df.head()`, `df.info()`, `df.describe()` for structure and stats
- Checked duplicate rows and dataset shape

### 3. Exploratory Data Analysis (EDA)
- Line plots for `CRIM`, `LSTAT`, `MEDV` to observe trends
- Histogram of `CRIM` distribution
- Seaborn `distplot` for `CRIM`
- Scatter plots: `CRIM` vs `MEDV`, `ZN` vs `MEDV`
- KDE plot: `CHAS` vs `MEDV`

### 4. Feature Selection
- Dropped `ZN` (weak relationship with target, seen via scatterplot)
- Dropped `CHAS`, `RAD`, `AGE`, `INDUS` — considered less useful for modeling

### 5. Train-Test Split
- `X` = all features except `MEDV`; `y` = `MEDV`
- `test_size = 0.3`, `random_state = 2`

### 6. Distribution Check (Q-Q Plots)
- Custom `qq_plot(feature)` function — combines histogram + Q-Q plot to visually check normality of a feature (e.g., `NOX`)

### 7. Feature Transformation
- Applied **PowerTransformer (Yeo-Johnson method)** to reduce skewness and make features more Gaussian
- Yeo-Johnson chosen over Box-Cox since it supports both positive and negative values
- Re-checked distribution with Q-Q plot after transformation

### 8. Model Building & Evaluation

| Model | Notes |
|---|---|
| **Linear Regression (baseline)** | Trained on original (untransformed) data |
| **Linear Regression (transformed)** | Trained on Power-Transformed data — compared MSE & R² against baseline |
| **Random Forest Regressor** | `n_estimators=400`, `max_depth=10` — trained on transformed data |
| **Gradient Boosting Regressor** | Advanced ensemble, learns from residual errors |

- **Evaluation metrics:** Mean Squared Error (MSE) and R² Score
- **Feature Importance:** extracted from Random Forest and plotted as horizontal bar chart

### 9. Hyperparameter Tuning
- Used **GridSearchCV** (cv=5, scoring='r2') on Random Forest
- Tuned `n_estimators`: [200, 300, 400] and `max_depth`: [8, 10, 15]

### 10. Log Transformation of Target (Advanced Trick)
- Applied `np.log1p()` on `y_train`/`y_test` to handle target skewness
- Trained Gradient Boosting on log-transformed target
- Inverse-transformed predictions using `np.expm1()` before evaluating R²

### 11. Output
Final processed dataset exported as `housing_price.csv`

---

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** VS Code + Jupyter Notebook (Anaconda)
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization
  - `scipy.stats` — Q-Q plot / normality checks
  - `scikit-learn` — preprocessing (PowerTransformer), modeling, GridSearchCV, evaluation metrics

---

## 📁 Repository Structure (suggested)
│
├── housing.csv # raw dataset
├── housing_price_.ipynb # main notebook (EDA + modeling)
└── README.md


---

## 🚀 How to Run

1. Install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
2. Place `housing.csv` in the project root.
3. Open the notebook in VS Code / Jupyter and run all cells sequentially.

---

## 📈 Key Takeaways

- `RM` and `LSTAT` are the strongest predictors of house price.
- Power Transformation (Yeo-Johnson) improved feature normality, aiding linear model performance.
- Ensemble models (Random Forest, Gradient Boosting) outperformed plain Linear Regression.
- Log-transforming a skewed target variable is a useful trick to stabilize model training.

---

## 🔮 Future Improvements

- Reconsider dropped features (`RAD`, `AGE`, `INDUS`) with proper correlation analysis instead of dropping outright
- Add cross-validation scores for Gradient Boosting, not just a single train/test split
- Try XGBoost / LightGBM for further performance gains
- Add residual plots to diagnose model errors visually
