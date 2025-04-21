# spring-2025-greenlight


# Movie Box Office Success Predictor

> **A data science approach to predicting blockbusters before they happen**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Data](https://img.shields.io/badge/movies-6600%2B-yellow.svg)](data/)

## Project Overview

This project builds a data-driven decision system that predicts whether a movie will succeed at the box office based on key influential factors. Our models identify patterns that distinguish financially successful films from unsuccessful ones, providing valuable insights for industry professionals and film enthusiasts alike.

## Objectives

While box office success is multi-dimensional (encompassing popularity, critical acclaim, cultural impact), our focus is primarily on **financial performance**. We've identified Return on Investment (ROI) as our primary success metric, with total gross revenue as a secondary metric when budget data is unavailable.

### Challenges Addressed:

- Collection and integration of diverse movie datasets
- Handling missing production budget information (typically not publicly available)
- Creating robust feature engineering for complex film variables 
- Building models that capture non-linear relationships between movie attributes and financial outcomes

## Repository Structure

Our repository is organized to reflect our end-to-end data science workflow:

### 📔 Notebooks

| Notebook | Description |
|----------|-------------|
| **`Import_Data.ipynb`** | Merges raw datasets from multiple sources into a unified `initial_build.pkl` for subsequent analysis |
| **`Data_Cleaning.ipynb`** | Comprehensive data cleaning procedures producing the `clean_build.pkl` dataset |
|**`engineer_features.ipynb`**| This Notebook contains all the feature engineering methods we used, and finally creates the `analysis_build.pkl` to be used in the next stage of the analysis.|
| **`Webscraping_MissingValues.ipynb`** | Implements scraping strategies to retrieve missing values for critical fields from open data sources |
| **`Comprehensive_MLPipeline.ipynb`** | Complete execution of our machine learning pipeline with model development, evaluation, and interpretation |
|**`build_regression_models.ipynb`** |      |

### 🐍 Python Modules

| Module | Description |
|--------|-------------|
| **`model_pipeline.py`** | Core implementation of model classes used in the ML pipeline (imported by `Comprehensive_MLPipeline.ipynb`) |

### 📁 Data Directories

| Directory | Description |
|-----------|-------------|
| **`raw_data/`** | Contains all the original datasets collected from different sources. |
| **`processed_data/`** | Houses all intermediate and final processed datasets created throughout the analysis pipeline, including cleaned, merged, and feature-engineered datasets |


## Data Sources & Features

Our analysis incorporates data from multiple sources, including:

- Production budgets for 6,600+ movies (scraped from public sources)
- Box office revenue (domestic and worldwide)
- Film attributes (genre, runtime,  rating, etc.)
- Creative team information (directors, actors, producers)
- Release strategy variables (season, competition, screens)

## Overview of the ML pipeline

Our strategy employs a four-stage workflow built on multiple base models (RandomForest, XGBoost, LightGBM, CatBoost, SVR, Lasso, and KNN) to capture diverse patterns in film data. After Optuna-driven hyperparameter optimization using time-series cross-validation, the system leverages a stacked ensemble architecture where base models generate chronologically-sound predictions that become features for a Ridge Regression meta-learner. This meta-model learns optimal weights for each algorithm's contribution, effectively combining their strengths while maintaining strict temporal boundaries to prevent future data leakage. Throughout this process, we also calculate SHAP values to quantify feature importance.

![mermaid-ai-diagram-2025-04-19-045753](https://github.com/user-attachments/assets/45a249a4-dda6-4179-895c-6870738d7611)

Key features:

### Time-Series Integrity
- **Chronological splitting** of data by release date
- **TimeSeriesSplit** cross-validation respecting temporal boundaries
- **Fold-isolated feature scaling** to prevent data leakage

### Ensemble Architecture
- **Out-of-fold (OOF) prediction strategy** for stacked modeling
- Base models trained on past data predict future-only validation sets
- Meta-model combines predictions using Ridge regression

### Model Support
- Tree-based (RandomForest, XGBoost, LightGBM, CatBoost)
- Linear and non-parametric (Ridge, Lasso, KNN, SVR)

### Optimization & Interpretability
- **Optuna-based hyperparameter tuning** with time-aware validation
- **feature importance using SHAP** 

![output](https://github.com/user-attachments/assets/499dab73-3a66-48b5-998c-1f164dbc0c6b)

## Key Findings

- Director rating performance indicates that directors with stronger track records positively influence predictions. 

- Horror films also demonstrate positive effects, likely due to high ROI relative to their typically lower budgets, and Family films show a substantial positive impact on box office predictions.

- Both very high budgets and very low budgets can have a positive impact on predictions.

- Longer films consistently contribute to higher predicted box office, but this effect diminishes over time.
  
- Lead actor efficiency (return on investment) shows notable importance in driving positive predictions.



## References

* Quader, N., Chaki, D., Gani, M. O., & Ali, M. H. (2017). A Machine Learning Approach to Predict Movie Box-Office Success. *2017 20th International Conference of Computer and Information Technology (ICCIT)*, 1-6. IEEE.
* Hong, S., & Wei, X. (2025). Blockbuster or bust? Silver screen effect and stock returns. *Review of Finance*, 1-30. [DOI: 10.1093/rof/rfaf004](https://doi.org/10.1093/rof/rfaf004)
* Apala, K. R., Jose, M., Motnam, S., Chan, C. C., Liszka, K. J., & de Gregorio, F. (2013). Prediction of Movies Box Office Performance Using Social Media. *2013 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining*, 1209-1214.
* Zhang, Z., Meng, Y., & Xiao, D. (2024). Prediction techniques of movie box office using neural networks and emotional mining. *Scientific Reports*, 14, 21209. [DOI: 10.1038/s41598-024-72340-z](https://doi.org/10.1038/s41598-024-72340-z)
* Is There a Right Way to Use AI to Make Movies? [https://open.spotify.com/episode/0gYjKq2fQDOZykEkg72L6t?si=jWg68PcKR7a30HsTBVDCCA](https://open.spotify.com/episode/0gYjKq2fQDOZykEkg72L6t?si=jWg68PcKR7a30HsTBVDCCA)

