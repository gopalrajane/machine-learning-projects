# Titanic Survival Prediction

A binary classification project that predicts passenger survival using the Titanic passenger dataset.

## Workflow

- Inspect and clean passenger data
- Handle missing values
- Perform exploratory data analysis
- Engineer features from passenger, ticket, and family information
- Train and compare classification models
- Evaluate model performance using cross-validation

## Final Features

The deployment model uses passenger class, sex, age, fare, and family size.

## Files

- `titanic_solve.ipynb` — data analysis and model experiments
- `titanic_train.csv` — training dataset
- `Profile_Report_titanic.html` — exploratory data analysis report
- `deployment/` — Flask and Streamlit prediction applications

## Run the Notebook

Open `titanic_solve.ipynb` in Jupyter Notebook or VS Code and install the required Python libraries.

## Run the Flask App

```bash
cd deployment
pip install -r requirements.txt
python app.py
```

## Run the Streamlit App

```bash
cd deployment
pip install -r requirements.txt
streamlit run streamlit_app.py
```
