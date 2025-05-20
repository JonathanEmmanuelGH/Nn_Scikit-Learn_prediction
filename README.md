# Neural Network Regression on Order Data

This project demonstrates the use of a shallow neural network (LeakyReLU + Dense layers) to predict a numerical target (`a6`) based on order-level data with date and categorical features. It also includes exploratory time-based visualizations like daily order volume and average orders per month.
Due to nda meassures the database has been cleaned and censored previously, the information displayed is only an altered information of the real one.
Also the main purpose of this files is to display the achievment on the elaboration of algorithms beyond the results. Now talking about them it is obvious that this is not the best algorithm to work on the database, if you like to see different approach please access to the different repositories in this profile.


## Project Structure

- `nn_scikit-learn_prediction.py` – main Python script that loads, processes, trains, evaluates, and visualizes the model and order data
- `jonathan_dataframe.parquet` – input data file (must be provided locally)
- `requirements.txt` – all required dependencies
- `README.md` – project documentation


## Model Summary

- **Architecture**: Sequential model with two hidden Dense layers (LeakyReLU)
- **Loss**: Mean Squared Error (MSE)
- **Target Scaling**: `StandardScaler`
- **Features**:
  - One-hot encoded categorical variables
  - Binary flags
  - Sine/Cosine cyclical encoding for date components


## Visualizations

- **Orders per day**  
  _Time series of order counts_
- **Average orders per month**  
  _Aggregated across years_
- **True vs. Predicted Scatter Plot**  
  _Filtered for clarity (values < 4000)_
- **Residual Plot**  
  _Error analysis for filtered predictions_


## Metrics (Sample)

Metric | Value (example) 

R²     | 0.51            
MAE    | ~0.19           
MSE    | ~0.53           

> _Note: these depend on the input dataset and training configuration._
> Due to results of this code it seems obvious that this is not the best model for use in the database


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/JonathanEmmanuelGH//Nn_Scikit-Learn_prediction.git
   cd Nn-Scikit-Learn-prediction
