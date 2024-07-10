import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pickle

# scikit-learn==1.5.1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 


# df = pd.read_csv("dataset\yellow_tripdata_2019-04.csv")
df = pd.read_csv("dataset\yellow_tripdata_2019-04-tiny.csv")

# QA
def questionA(df):
    # List of columns to calculate percentiles on
    value_columns = ['fare_amount', 'tip_amount', 'total_amount']
    percentiles = [5, 50, 95]
    
    # Group by the specified fields
    group_fields = ['VendorID', 'passenger_count', 'payment_type']

    # Prepare a dictionary to collect results
    results = {}
    for group_name in group_fields:
        grouped = df.groupby(group_name)
            
        # Iterate over each group and index
        for name, group in grouped:
            # Calculate percentiles for each column in value_columns
            for col in value_columns:
                for p in percentiles:
                    # Percentile column name
                    col_name = f"{col}_p_{p}"
                    # Calculate percentile
                    value = group[col].quantile(p / 100.0)
                    # Add the result to the dictionary
                    key = f"{group_name}_{int(name)}"

                    if key not in results:
                        results[key] = {}
                        print(key)
                    results[key][col_name] = value
    
    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df

# QA1
def questionA1(df):
    # List of columns to calculate percentiles on
    value_columns = ['fare_amount', 'tip_amount', 'total_amount']
    percentiles = [5, 50, 95]
    
    # Prepare a dictionary to collect results
    results = {}
    # Group by trip_distance
    df.loc[df.trip_distance > 2.8, 'trip_distance_cond'] = 1
    df.loc[df.trip_distance <= 2.8, 'trip_distance_cond'] = 0
    grouped = df.groupby('trip_distance_cond')
        
    # Iterate over each group
    for name, group in grouped:
        # Calculate percentiles for each column in value_columns
        for col in value_columns:
            for p in percentiles:
                # Percentile column name
                col_name = f"{col}_p_{p}"
                # Calculate percentile
                value = group[col].quantile(p / 100.0)
                # Add the result to the dictionary
                if int(name) == 0: key = f"trip_distance<=2.8"
                else: key = f"trip_distance>2.8"

                if key not in results:
                    results[key] = {}
                    print(key)
                results[key][col_name] = value

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df

# QA and QA1 together
def questionAA1(df):
    # List of columns to calculate percentiles on
    value_columns = ['fare_amount', 'tip_amount', 'total_amount']
    percentiles = [5, 50, 95]
    
    # Group by the specified fields
    group_fields = ['VendorID', 'passenger_count', 'payment_type', 'trip_distance_cond']

    df.loc[df.trip_distance > 2.8, 'trip_distance_cond'] = 1
    df.loc[df.trip_distance <= 2.8, 'trip_distance_cond'] = 0
    grouped = df.groupby('trip_distance_cond')

    # Prepare a dictionary to collect results
    results = {}
    for group_name in group_fields:
        grouped = df.groupby(group_name)
            
        # Iterate over each group and index
        for name, group in grouped:
            # Calculate percentiles for each column in value_columns
            for col in value_columns:
                for p in percentiles:
                    # Percentile column name
                    col_name = f"{col}_p_{p}"
                    # Calculate percentile
                    value = group[col].quantile(p / 100.0)
                    # Add the result to the dictionary
                    if group_name == "trip_distance_cond":
                        if int(name) == 0: key = f"trip_distance<=2.8"
                        else: key = f"trip_distance>2.8"
                    else: key = f"{group_name}_{int(name)}"

                    if key not in results:
                        results[key] = {}
                        print(key)
                    results[key][col_name] = value
    
    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df

# QB
def questionB(df):
    x_name = ['VendorID', 'passenger_count', 'payment_type', 'trip_distance']
    y_name = 'total_amount'
    df = df[['VendorID', 'passenger_count', 'payment_type', 'trip_distance', 'total_amount']]
    df = df.replace(np.nan, 0) 
    # df=(df-df.mean())/df.std() # mean / std normalization
    # df=(df-df.min())/(df.max()-df.min()) # min / max normalization

    # Dividing dataset into TRAIN and TEST sets
    x, y = df[x_name], df[y_name]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

    # Applying LINEAR REGRESSION
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # RMSE Calculation
    rmse_linr = sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE Linear Regression: {rmse_linr}") # RMSE 0+ (smaller is better)
    r2_linr = r2_score(y_test, y_pred)
    print(f"R2 Linear Regression: {r2_linr}") # R Squared 0-1 (larger is better)
    
    plt.plot(y_pred, y_test, 'o', color='blue', markersize=0.5)
    plt.plot(y_test, y_test, 'o', color='red', markersize=0.5)
    plt.axis((0, 100, 0, 100))

    plt.show()

    return model

print()

# df_res_A = questionA(df)
# df_res_A.to_csv('res/answerA.csv', index=True)
# df_res_A1 = questionA1(df)
# df_res_A1.to_csv('res/answerA1.csv', index=True)
df_res_AA1 = questionAA1(df)
df_res_AA1.to_csv('res/answerAA1.csv', index=True)
model = questionB(df)
with open('res/ml.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
