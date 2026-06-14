# House Price Predictor

A machine learning project that predicts house prices based on various property features using regression models.

## Overview

This project uses historical housing data to train machine learning models and estimate house prices for new properties. The workflow includes:

* Data loading and preprocessing
* Handling missing values
* Feature engineering
* Model training
* Model evaluation
* Price prediction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Models Used

* Linear Regression
* Random Forest Regressor

## Project Structure

```
house-price-predictor/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook or Python script to:

1. Load the dataset
2. Preprocess the data
3. Train the model
4. Evaluate performance
5. Generate predictions

## Evaluation Metric

The models are evaluated using:

* Mean Absolute Error (MAE)

Lower MAE indicates better predictive performance.

## Results

Model Comparison

Linear Regression
- MAE: 18,409

Random Forest Regressor
- MAE: 17,528
- 5-Fold CV MAE: 17,638

XGBoost Regressor
- MAE: 16,032
- 5-Fold CV MAE: 15,983

Best Model: XGBoost

## Future Improvements

* Hyperparameter tuning
* Feature engineering
* XGBoost implementation
* Model deployment using Flask or Streamlit
* Real-time prediction interface

## License

This project is open-source and available under the MIT License.
