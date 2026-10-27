Titanic: Survival Prediction

A machine learning project that predicts the survival probability of a Titanic passenger, with an interactive web interface built using **Streamlit**.

The model is a scikit-learn pipeline (preprocessing + Logistic Regression) trained on the classic Titanic dataset. The app takes passenger details as input and returns a survival probability.

## Features

- Predicts survival probability from passenger class, sex, age, fare, family size, title and port of embarkation.
- Simple, form-based web interface (Streamlit); the UI language is Uzbek.
- Full data analysis and model training workflow in a Jupyter Notebook.
- Preprocessing and model are saved together as one pipeline, so raw input can be passed directly to `predict_proba`.

## Project Structure

```
titanic-ml-project/
├── app.py                 # Streamlit web application
├── Titanic.ipynb          # EDA, preprocessing, model comparison and training
├── titanic_model.joblib   # Trained pipeline (preprocessing + Logistic Regression)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Dataset

- Source: [Titanic dataset (Data Science Dojo)](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
- 891 passengers, 12 columns; target column is `Survived` (0 = no, 1 = yes).
- Class balance: 549 did not survive, 342 survived.
- Missing values: `Cabin` (687), `Age` (177), `Embarked` (2).

## Methodology

**Feature engineering**
- `FamilySize = SibSp + Parch + 1`
- `Title` extracted from `Name` (Mlle/Ms → Miss, Mme → Mrs); anything other than Mr, Mrs, Miss, Master is grouped as `Rare`.
- Dropped columns: `PassengerId`, `Name`, `Ticket`, `Cabin`.

**Preprocessing (scikit-learn `ColumnTransformer`)**
- Numeric (`Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`): median imputation, then `StandardScaler`.
- Categorical (`Pclass`, `Sex`, `Embarked`, `Title`): most-frequent imputation, then `OneHotEncoder`.

**Training**
- Stratified 80/20 train/test split (`random_state=42`).
- Several models were compared on the same pipeline, and the best one was tuned with `GridSearchCV` (5-fold, ROC-AUC).

## Results

Model comparison on the test set (179 passengers):

| Model               | Accuracy | F1    | ROC-AUC |
|---------------------|----------|-------|---------|
| Logistic Regression | 0.838    | 0.779 | 0.878   |
| KNN                 | 0.816    | 0.748 | 0.855   |
| SVM                 | 0.838    | 0.782 | 0.853   |
| Random Forest       | 0.821    | 0.758 | 0.832   |

Tree-based models (with limited depth) were also tested: Random Forest (ROC-AUC 0.852), Gradient Boosting (0.851), AdaBoost (0.847) and Decision Tree (0.838). None of them beat Logistic Regression.

**Final model:** Logistic Regression with `C=10` (best 5-fold CV ROC-AUC: 0.866).

| Metric   | Test score |
|----------|------------|
| Accuracy | 0.838      |
| F1       | 0.782      |
| ROC-AUC  | 0.882      |

> The test set is small (179 samples) and a single split was used, so these numbers can vary by a few percentage points with a different split.

## Installation

Requirements: Python 3.9+ (tested with 3.13)

```bash
# 1. Clone the repository
git clone https://github.com/muxtorov01/titanic-ml-project.git
cd titanic-ml-project

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**Run the web app**

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501), fill in the passenger form and click **Bashorat qilish** (Predict).

**Use the model in Python**

```python
import joblib
import pandas as pd

model = joblib.load("titanic_model.joblib")

passenger = pd.DataFrame([{
    "Pclass": 1, "Sex": "female", "Age": 30.0,
    "SibSp": 0, "Parch": 0, "Fare": 80.0,
    "Embarked": "S", "FamilySize": 1, "Title": "Miss",
}])

print(model.predict_proba(passenger)[0, 1])   # survival probability
```

**Retrain the model**

Open `Titanic.ipynb` in Jupyter or Google Colab and run all cells. The last training cell saves a new `titanic_model.joblib`.

## Input Features

| Feature      | Description                                  | Values                          |
|--------------|----------------------------------------------|---------------------------------|
| `Pclass`     | Ticket class                                 | 1, 2, 3                         |
| `Sex`        | Sex                                          | `male`, `female`                |
| `Age`        | Age in years                                 | 0–80                            |
| `SibSp`      | Number of siblings/spouses aboard            | 0–8                             |
| `Parch`      | Number of parents/children aboard            | 0–6                             |
| `Fare`       | Ticket fare                                  | 0–520                           |
| `Embarked`   | Port of embarkation                          | `S`, `C`, `Q`                   |
| `FamilySize` | `SibSp + Parch + 1` (computed in the app)    | integer                         |
| `Title`      | Title from the passenger's name              | Mr, Mrs, Miss, Master, Rare     |

## Important Note on Versions

`titanic_model.joblib` was saved with **scikit-learn 1.6.1**. Loading it with a different scikit-learn version may raise an error or give unreliable results, so keep the version pinned in `requirements.txt`, or retrain the model with your own version using the notebook.

## Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib / Seaborn (for the notebook visualizations)

## Disclaimer

This project was created for educational purposes. Predictions are statistical estimates based on historical data and are not meant for any real-world decision making.

## Author

[@muxtorov01](https://github.com/muxtorov01)
