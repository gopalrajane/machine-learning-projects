# 🚢 Titanic Survival Prediction

A complete end-to-end **binary classification** project that predicts whether a passenger survived the Titanic disaster, based on demographic and travel information. The notebook covers data cleaning, exploratory data analysis (EDA), feature engineering, and comparison of five different machine learning models.

---

## 📌 Problem Statement

Given passenger-level data (age, sex, class, fare, family size, cabin, ticket, etc.), predict the binary target variable **`Survived`** (0 = Did not survive, 1 = Survived).

---

## 📊 About the Dataset

The dataset used is the well-known **Titanic dataset** (`titanic_train.csv`), containing one row per passenger.

| Column | Description |
|---|---|
| `PassengerId` | Unique identifier for each passenger *(dropped — no predictive value)* |
| `Survived` | Target variable — 0 = No, 1 = Yes |
| `Pclass` | Ticket class (1 = Upper, 2 = Middle, 3 = Lower) |
| `Name` | Passenger's full name *(used to engineer `Title`, then dropped)* |
| `Sex` | Gender of the passenger |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number *(used to engineer ticket-based features, then dropped)* |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number *(used to engineer deck/cabin features, then dropped)* |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🔄 Project Workflow

### 1. Data Loading
- Libraries used: `pandas`, `numpy`, `sklearn.impute.SimpleImputer`
- Dataset loaded via `pd.read_csv('titanic_train.csv')`

### 2. Initial Cleaning
- Dropped `PassengerId` — it's just a unique identifier with no predictive signal.
- Checked dataset shape and missing values (`df.isnull().sum()`).

### 3. Handling Missing Values
- **`Cabin`** → missing values filled with `0` (placeholder, since most cabin data is missing).
- **`Age`** → missing values filled with the **mean age**.
- **`Embarked`** → missing values filled using **most frequent value** via `SimpleImputer(strategy='most_frequent')`.

### 4. Exploratory Data Analysis (EDA)
- Conditional counting via a custom `value()` function (e.g., how many survived by `Sex`, `Pclass`, `Parch`).
- Visualizations using **Seaborn**:
  - Count plot of overall survival
  - Passenger class distribution
  - Survival by passenger class (`hue='Survived'`)
- Custom `bar_graph(feature)` function — compares Survived vs Dead counts across any categorical feature (used for `Sex`, `Pclass`, `Embarked`, `Parch`).
- **Automated EDA report** generated using `ydata_profiling.ProfileReport`, exported as `Profile_Report_titanic.html`.

### 5. Feature Engineering
- **`Title`** — extracted from `Name` (e.g., Mr, Mrs, Miss) using regex; `Name` column then dropped.
- **`Ticket_num` & `Ticket_cat`** — `Ticket` split into a numeric part and a category prefix.
- **`Family`** — new feature = `SibSp` + `Parch` + 1 (total family size aboard).
- **`Sex` encoding** — mapped to numeric (`male` → 0, `female` → 1).
- **`Deck_Floor` & `Cabin_Number`** — extracted from `Cabin` using regex (letter = deck, digits = cabin number).

### 6. Feature Selection
Dropped columns that were either redundant or already captured by engineered features:
`SibSp`, `Parch`, `Cabin`, `Title`, `Ticket_num`, `Ticket_cat`, `Deck_Floor`, `Cabin_Number`, `Embarked`.

Final feature set kept: `Pclass`, `Sex`, `Age`, `Fare`, `Family`.

### 7. Train-Test Split
- `X` = all features except `Survived`; `y` = `Survived`
- `test_size = 0.3` (70% train / 30% test), `random_state = 42` for reproducibility

### 8. Model Building & Evaluation
Five classification models were trained and evaluated using **10-fold cross-validation** (`KFold(n_splits=10, shuffle=True, random_state=0)`):

| # | Model | Notes |
|---|---|---|
| 1 | **K-Nearest Neighbors (KNN)** | Tuned `k` by testing values 1–30 to find the best-performing `k` |
| 2 | **Decision Tree** | Default parameters, cross-validated |
| 3 | **Random Forest** | Tested with varying `n_estimators` |
| 4 | **Gaussian Naive Bayes** | Probabilistic baseline model |
| 5 | **Support Vector Machine (SVM)** | Default `SVC()` |

### 9. Model Comparison
All five models were re-run under a common evaluation loop, and their mean cross-validation accuracy scores were compiled into a **leaderboard DataFrame**, sorted from best to worst performing model.

### 10. Output
Final cleaned dataset exported as `titanic_cleaned.csv` for downstream use.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** VS Code + Jupyter Notebook (Anaconda)
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `seaborn` — visualization
  - `ydata_profiling` — automated EDA
  - `scikit-learn` — preprocessing, modeling, cross-validation, evaluation

---

## 📁 Repository Structure (suggested)

```
titanic-survival-prediction/
│
├── titanic_train.csv          # raw dataset
├── titanic_solve.ipynb        # main notebook (EDA + modeling)
├── titanic_cleaned.csv        # processed output dataset
├── Profile_Report_titanic.html # auto-generated EDA report
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy seaborn scikit-learn ydata-profiling
   ```
2. Place `titanic_train.csv` in the project root.
3. Open `titanic_solve.ipynb` in VS Code / Jupyter and run all cells sequentially.

---

## 📈 Key Takeaways

- Sex, passenger class, and fare emerged as strong visual indicators of survival during EDA.
- Feature engineering (Title, Family size, Deck info) added meaningful signal beyond raw columns.
- Model comparison via cross-validation gives a fair, leakage-free way to pick the best-performing algorithm before final testing.

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV) for top-performing model
- Add confusion matrix / precision-recall analysis on the held-out test set
- Try ensemble methods like XGBoost or stacking
- Handle `Ticket_cat`, `Deck_Floor` more robustly instead of dropping (potential signal loss)
