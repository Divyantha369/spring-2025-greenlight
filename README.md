# spring-2025-greenlight


![mermaid-ai-diagram-2025-04-19-045753](https://github.com/user-attachments/assets/45a249a4-dda6-4179-895c-6870738d7611)


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
|**`engineer_features.ipynb`**| This Notebook contains all the feature engineering methods we used|
| **`Webscraping_MissingValues.ipynb`** | Implements scraping strategies to retrieve missing values for critical fields from open data sources |
| **`Comprehensive_MLPipeline.ipynb`** | Complete execution of our machine learning pipeline with model development, evaluation, and interpretation |
|**`build_regression_models.ipynb`** |      |

### 🐍 Python Modules

| Module | Description |
|--------|-------------|
| **`model_pipeline.py`** | Core implementation of model classes used in the ML pipeline (imported by `Comprehensive_MLPipeline.ipynb`) |

## Data Sources & Features

Our analysis incorporates data from multiple sources, including:

- Production budgets for 6,600+ movies (scraped from public sources)
- Box office revenue (domestic and worldwide)
- Film attributes (genre, runtime,  rating, etc.)
- Creative team information (directors, actors, producers)
- Release strategy variables (season, competition, screens)

## Key Findings

Our analysis reveals several critical factors that significantly impact box office success:

![output](https://github.com/user-attachments/assets/499dab73-3a66-48b5-998c-1f164dbc0c6b)
