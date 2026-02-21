# Real estate price prediction

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Train

1. Put your scraped CSV at: data/raw/listings.csv
2. Run:

   python run_train.py

To skip geocoding:

   python run_train.py --no-geo

Artifacts:
- models/rf_pipeline.joblib
- models/xgb_pipeline.joblib
- reports/metrics.json
- reports/*.png

## Predict

1. Prepare an input CSV with the same feature columns.
2. Run:

   python run_predict.py --input data/raw/new_listings.csv

Optional:
- Choose a model:

   python run_predict.py --model models/rf_pipeline.joblib --input data/raw/new_listings.csv

To skip geocoding:

   python run_predict.py --input data/raw/new_listings.csv --no-geo
