# Demand-Aware Optimization for Dynamic Bike-Sharing Rebalancing

## Objective
Build an end-to-end framework for bike-sharing rebalancing combining:

- Station-level demand forecasting
- Vehicle routing optimization
- Simulation of rebalancing strategies

## Project Pipeline

1. Data collection and preprocessing
2. Feature engineering and demand forecasting
3. Optimization modeling (MILP routing problem)
4. Rolling-horizon simulation

## Tech Stack

- Python (pandas, numpy)
- scikit-learn / XGBoost
- OR-Tools / Gurobi
- Streamlit

## Getting Started

Follow the steps below to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/<your-organization-or-username>/bike-sharing-rebalancing.git
cd bike-sharing-rebalancing
```

### 2. Install dependencies

We recommend using a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add the data

The datasets are not included in the repository.  
Obtain the data from **Martyna or Pierre** and place the files in the following folder:

```
data/raw/
```

Expected structure:

```
data
├── raw
│   ├── Divvy_Trips_2017_Q1.csv
│   ├── Divvy_Trips_2017_Q2.csv
│   ├── Divvy_Trips_2017_Q3.csv
│   ├── Divvy_Trips_2017_Q4.csv
│   └── Divvy_Stations_2017_Q3Q4.csv
└── processed
```

### 4. Build the dataset

Run the data pipeline to clean the data, construct the hourly demand dataset, and add weather features.

```bash
python src/data/build_dataset.py
```

The processed dataset will be saved to:

```
data/processed/
```

## Repository Structure

- `data/` raw and processed datasets
- `src/` core code
- `notebooks/` exploration
- `results/` figures and metrics
