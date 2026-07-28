# 🏘️ House Price Prediction (Advanced Regression)

An end-to-end **regression** project predicting `SalePrice` of residential homes using the Kaggle "House Prices - Advanced Regression Techniques" style dataset (`train.csv` / `test.csv`). Covers deep missing-value imputation, categorical encoding (one-hot + ordinal), outlier capping across 15+ numerical features, and comparison of six regression models.

---

## 📌 Problem Statement

Given 79+ explanatory variables describing residential homes (size, quality, location, condition, garage, basement, etc.), predict the target variable **`SalePrice`** — the final sale price of each house.

---

## 📊 About the Dataset

Standard **train/test split** provided as separate files (`train.csv` has the target `SalePrice`; `test.csv` does not — used for final predictions). Dataset has a mix of numerical and categorical features covering:

- **Identification:** `Id` *(dropped — no predictive value)*
- **Zoning & Lot:** `MSSubClass`, `MSZoning`, `LotFrontage`, `LotArea`, `LotShape`, `LotConfig`
- **Location:** `Neighborhood`, `Condition1`, `Condition2`, `Street`, `LandContour`, `LandSlope`
- **Building type/style:** `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `Foundation`
- **Quality/Condition ratings:** `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `Functional`
- **Basement details:** `BsmtExposure`, `BsmtFinType1/2`, `BsmtFinSF1/2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`
- **Areas:** `1stFlrSF`, `2ndFlrSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`
- **Utilities & systems:** `Utilities`, `Heating`, `CentralAir`, `Electrical`, `PavedDrive`
- **Sale info:** `SaleType`, `SaleCondition`
- **Target:** `SalePrice`

Columns with very high missingness (`Alley`, `PoolQC`, `Fence`, `MiscFeature` — each missing 1100+ rows in both train/test) were dropped early, along with `Id`.

---

## 🔄 Project Workflow

### 1. Data Loading
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Loaded `train.csv` and `test.csv` separately
- Configured pandas to display all rows/columns for easier inspection

### 2. Initial Inspection
- `.head()`, `.columns`, `.shape`, `.info()`, `.describe()`
- Checked missing values in both train and test independently

### 3. Removing Highly Missing Features
- Dropped `Id`, `Alley`, `PoolQC`, `Fence`, `MiscFeature` (extremely high missing counts, low predictive value)

### 4. Missing Value Imputation
- **Categorical "None" fill** (feature genuinely absent, e.g. no basement/garage/fireplace): `MasVnrType`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`
- **Most-frequent imputation** (`SimpleImputer(strategy='most_frequent')`): `SaleType`, `MSZoning`, `Electrical`, `Utilities`, `Exterior1st`, `Exterior2nd`, `KitchenQual`, `Functional`
- **Median imputation:** `LotFrontage` (compared mean vs. median fill visually using KDE plots before choosing median)
- **Zero-fill** (numeric fields tied to "no such feature"): `GarageCars`, `GarageArea`, `GarageYrBlt`, `MasVnrArea`, `BsmtFullBath`, `BsmtHalfBath`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`
- All imputers fit on `train` and applied (`.transform`) to `test` to avoid data leakage

### 5. Categorical Encoding

**One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) applied to nominal (unordered) categorical features:
`MSSubClass`, `MSZoning`, `Street`, `LandContour`, `LotConfig`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `Foundation`, `Heating`, `CentralAir`, `Electrical`, `GarageType`, `SaleType`, `SaleCondition`
— original columns dropped after encoding to avoid duplication.

**Ordinal Encoding** (manual mapping dictionaries preserving rank order) applied to quality/condition-style features:
`LotShape`, `Utilities`, `LandSlope`, `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `HeatingQC`, `KitchenQual`, `Functional`, `FireplaceQu`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`
— e.g. `Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0`.

### 6. Outlier Detection & Handling
Systematically checked and capped outliers (before/after distribution + boxplot comparison) across 15 numerical features:
`LotFrontage` *(Z-score / 3-sigma capping)*, `LotArea`, `YearBuilt`, `MasVnrArea`, `BsmtFinSF1`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch` *(IQR method — 1.5×IQR capping, computed separately for train and test)*
- Reusable `plot(feature)` helper function generates a 4-panel view (train dist, test dist, train box, test box) for quick before/after comparison

### 7. Train/Test Feature Preparation
- Separated `X` (features) and `y` (`SalePrice`) from `train`
- Used `pd.get_dummies()` for a full one-hot pass and `X.align(X_test, join='left', axis=1, fill_value=0)` to guarantee train/test columns match exactly

### 8. Train/Validation Split
- `train_test_split(X, y, test_size=0.2, random_state=42)` — 80% train / 20% validation for model evaluation

### 9. Model Building & Evaluation
Six regression models trained and evaluated on the validation set using **MAE, MSE, RMSE, and R² Score**:

| # | Model | Notes |
|---|---|---|
| 1 | **Linear Regression** | Baseline model |
| 2 | **Decision Tree Regressor** | `random_state=42` |
| 3 | **Random Forest Regressor** | `n_estimators=100`, `random_state=42` |
| 4 | **XGBoost Regressor** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6` |
| 5 | **Gradient Boosting Regressor** | Default params, `random_state=42` |
| 6 | **AdaBoost Regressor** | Default params, `random_state=42` |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** VS Code + Jupyter Notebook (Anaconda)
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization (distributions, boxplots)
  - `scikit-learn` — imputation, encoding, modeling, evaluation metrics
  - `xgboost` — gradient boosted tree regression

---

## 📁 Repository Structure (suggested)