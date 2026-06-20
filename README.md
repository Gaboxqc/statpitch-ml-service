# StatPitch

A complete machine learning pipeline designed to predict international football match outcomes and power standard betting markets (1X2, Over/Under, Both Teams to Score, and Correct Score). 

This project is broken down into a series of Jupyter Notebooks that handle everything from raw data ingestion to feature engineering and ensemble model training, with the ultimate goal of deploying the predictive engine via a REST API.

## Project Architecture

The pipeline consists of the following sequential stages:

### 1. Data Collection (`01_Data_Collection.ipynb`)
* **Automated Ingestion:** Fetches historical international football results directly from a public community-maintained GitHub repository (no API key required).
* **Filtering & Cleaning:** Focuses on modern football (1994–present), filtering for highly competitive matches including the FIFA World Cup, World Cup Qualifiers, friendlies, and major continental tournaments.
* **Target Creation:** Generates essential baseline metrics and target variables, such as total goals, goal differentials, and boolean flags for markets like BTTS (Both Teams To Score) and Over 2.5 goals.

### 2. Feature Engineering (`02_Feature_Engineering.ipynb`)
This stage builds 25 predictive features with a strict "no data leakage" policy, ensuring models only learn from data available *before* a match kicks off. Features are grouped into three categories:
* **Elo Ratings:** Calculates dynamic team strength that updates after every match, heavily weighting major tournaments. The Elo difference is the strongest predictor of match outcomes.
* **Rolling Form:** Captures team momentum by calculating the rolling averages of goals scored, goals conceded, and points per game over the last 5 and 10 matches.
* **Head-to-Head (H2H):** Analyzes the historical psychological advantage between two specific teams by reviewing their last 10 meetings.

### 3. Model Training (`03_Model_Training.ipynb`)
The predictive engine utilizes a "walk-forward" split, training on all matches before 2022 and strictly testing against the real-world 2022 FIFA World Cup to ensure honest accuracy metrics. 
* **XGBoost Classifier:** Learns the direct probability of a Home Win, Draw, or Away Win (1X2 market).
* **XGBoost-Poisson Regressor:** Two separate models estimate the expected goals for the home and away teams. These expected goals populate a probability matrix to power exact scores, Over/Under, and BTTS markets.
* **Ensemble Blending:** The final 1X2 prediction blends the XGBoost classifier probabilities (60%) with the Poisson-derived probabilities (40%) to create a highly accurate consensus. Models are serialized via `joblib` for production use.

### 4. Backend Deployment (Upcoming)
The serialized models (`.pkl`) and the configuration matrices will be wrapped into a backend service using **FastAPI**. This REST API will allow any client application to send two team names and instantly receive a full suite of match predictions and market probabilities.

## Tech Stack
* **Python 3**
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, SciKit-Learn, SciPy (Poisson distributions)
* **Backend:** FastAPI (Planned)
* **Frontend:** React (Planned)
* **Data Visualization:** Matplotlib

## Evaluation & Results
The models are evaluated against naive baselines (e.g., always predicting the most common outcome). The combined XGBoost and Poisson approach successfully outperforms these baselines across all major markets during the 2022 World Cup test set. 

## How to Run
1. Clone the repository.
2. Install the required dependencies: `pip install pandas numpy xgboost scikit-learn scipy matplotlib joblib fastapi`
3. Run the notebooks in chronological order (`01` through `03`) to generate the datasets, engineer the features, and output the trained `.pkl` models.