# ROBO-ADVISOR
# In case the Python API does not work, use the url with API key curl "https://data.nasdaq.com/api/v3/datatables/ETFG/FUND.csv?ticker=SPY&api_key=YOURAPIKEY"
# Website with all indicator API names https://data.nasdaq.com/data/
# nasdaqdatalink API manual https://github.com/Nasdaq/data-link-python/blob/main/FOR_ANALYSTS.md
# work with whats there.

# MACROECONOMIC INDICATORS
# 1. NOT LONGER AVAILABLE Consumer Sentiment - UMICH Consumer Survey Index (Consumer Sentiment) - "UMICH/SOC1" (once a month)
# 2. Unemployment - Federal Government; contributions for government social insurance and unemployment insurance - "FED/FA316231011_Q" (every quarter)
# 3. Interest Rates - Treasury Real Long-Term Rates - "USTREASURY/REALLONGTERM" (every trading day)
# 4. NOT LONGER AVAILABLE Inflation Rate - Consumer Price Index USA - "RATEINF/CPI_USA" (once a month)
# since indicators are all available new every quarter or less, shift indicators and project stock price for the next quarter

# TECHNICAL INDICATORS
# 1. Moving Average - can be calculated from yahoo finance
# 2. Volume Traded - yahoo finance
# 3. Momentum - can be calculated from yahoo finance

# COMPANY FUNDAMENTALS (SEC DATA)
# 1. Profit Margin
# 2. Return on Assets
# 3. Inventory Turnover
# 4. Current Ratio
# 5. Cash Ratio
# 6. Profit-Earnings Ratio


# FINAL IMPROVEMENTS TO BE IMPLEMENTED
# 1. Log returns
# 2. Momentum Benchmark 
# 3. ARIMA Modell Alternative statsmodel.tsa

# Download data 


### Parameters
# Company tickers that work: aapl, msft, fn, googl, tsla, unh, xom, jnj, pg, cvx, pep, lin, pfe, spgi, lmt, sbux, mdlz, cop, pm, txn, ibm, cat, rtx
# nee, syk, bmy, slb, qcom, nee, cvs 
company_ticker = "msft"
method = "a" # l = local (only msft) or a = online real-time api
last_days = 300 #how many trading days included
days_into_future = 21 #how many days into the future stock price projected



# Variable Stock name
name = str(company_ticker.upper() + " stock price")

### NASDAQDATALINK - Macroeconomic Data
import pandas as pd
if method == "a":
    key = "Y_D1VrAAHWRaqepWFzT3"
    data_start_date = "2015-01-01"
    import nasdaqdatalink
    
    data = nasdaqdatalink.get(["USTREASURY/REALLONGTERM","UMICH/SOC1","FED/FA316231011_Q",
                               "RATEINF/CPI_USA"], api_key=key, start_date = data_start_date)
    data = data.rename(columns={"USTREASURY/REALLONGTERM - LT Real Average (>10Yrs)":"10-year bonds", 
                                "UMICH/SOC1 - Index":"C. Sentiment", 
                                "FED/FA316231011_Q - Value":"Unemployment", 
                                "RATEINF/CPI_USA - Value":"Price Index"})
    data.fillna(method="ffill", inplace=True)
    data.fillna(method="bfill", inplace=True)
    
    
    ### YAHOO FINANCE - Stock price data
    
    import yahoo_fin.stock_info as si
    
    st_data = si.get_data(company_ticker, start_date = data_start_date)
    st_data = st_data[["adjclose","volume"]]
    
    ### SEC.GOV - Company Fundamentals
    
    import requests 
    
    
    headers = {"User-agent" : "sr2378@njit.edu"}
    companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    
    companyData = pd.DataFrame.from_dict(companyTickers.json(),orient="index")
    companyData["cik_str"] = companyData["cik_str"].astype(str).str.zfill(10)
    companyData = companyData.set_index("ticker")
    company_ticker = company_ticker.upper()
    cik = companyData.loc[company_ticker, "cik_str"]
    
    #confirmed ones
    # "Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent", "InventoryNet", --> no, too complicated: "GrossProfit"
    
    variable = "InventoryNet"
    companyConcept = requests.get((f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json'), 
                                  headers=headers)
    variableData = pd.DataFrame.from_dict((companyConcept.json()["units"]["USD"]))
    variableData = variableData.set_index("end")
    variableData = variableData[["val","form"]]
    variableData = variableData.sort_index()
    
    
    
    variables = ["Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent", "InventoryNet"]
    for variable in variables:
        companyConcept = requests.get(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json', 
                                      headers=headers)
        variableData = pd.DataFrame.from_dict(companyConcept.json()["units"]["USD"])
        variableData = variableData.set_index("end")
        variableData = variableData[["val", "form"]]
        variableData.columns = [f"{variable}_val", f"{variable}_form"]
        variableData = variableData.sort_index()
        variableData.reset_index(inplace=True)  # Reset the index
        variableData = variableData[variableData[f"{variable}_form"] == "10-Q"]
        variableData = variableData.drop_duplicates(subset=["end"])
        variableData = variableData[[f'{variable}_val', "end"]]
        variableData["end"] = pd.to_datetime(variableData["end"])
        variableData.set_index("end", inplace=True)
        exec(f'data_{variable} = variableData')
    
    first_part = variables[0]
    second_part = data_Liabilities, data_AssetsCurrent, data_LiabilitiesCurrent, data_InventoryNet
    
    merged_data = data_Assets
    
    for i in second_part:
        merged_data = pd.merge(merged_data, i, left_index = True, right_index = True, how="outer")
    
    
    
    ### Put Data together
    
    result = pd.merge(data, st_data, left_index=True, right_index=True, how="left")
    result = pd.merge(result, merged_data, left_index= True, right_index= True, how="left")
    result.fillna(method="ffill", inplace=True)
    result.fillna(method="bfill", inplace=True)
    result.rename(columns={"adjclose":str(company_ticker+" stock price")}, inplace=True)

elif method == "l":
        result = pd.read_excel("result_local.xlsx")
        result.set_index("Date",inplace=True)
        
else: 
    print("Select method 'local' or 'api'")
    
print(result)

# Drop all columns that we couldn't get data from
df = result.dropna(axis=1)


### Calculate Indicators
#Calculate 21-Day Moving Average

df["21-TradingDay (1 Month) Moving Average"] = df[name].rolling(window=21).mean()

#Calculate 21-Day Momentum
df["21-TradingDay (1 Month) Momentum"] = df[name].rolling(window=21).apply(lambda x: x.iloc[-1] - x.iloc[0])

#Calculate Assets/Liabilities Ratio
df["Assets/Liabilites Ratio"] = df["Assets_val"]/df["Liabilities_val"]
df.drop(columns=["Assets_val","Liabilities_val"], inplace = True)

#Calculate Current Ratio
df["Current Ratio"] = df["AssetsCurrent_val"] / df["LiabilitiesCurrent_val"]
df.drop(columns=["AssetsCurrent_val","LiabilitiesCurrent_val"], inplace= True)
df.drop(df.index[:20], inplace=True)


                

#next step: shift the stock data down by 21 spots (1 month in trading days)
df[name] = df[name].shift(-days_into_future)

# using only rows in which we have the stock price present
model_df = df.iloc[:-days_into_future]

# Shorten the data to only include recent trends

model_df = model_df[-last_days:]

# Standardize all x-values to 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_standardize = model_df.columns.tolist()
columns_to_standardize.remove(name)
model_df[columns_to_standardize] = scaler.fit_transform(model_df[columns_to_standardize])

### Create LogReturns from prices
import numpy as np
model_df[name] = np.log(model_df[name]) - np.log(model_df[name].shift(1))
model_df = model_df[1:]



x = model_df[columns_to_standardize]
y = model_df[name]



# Train Test Split (last 20% for testing)
count = len(model_df)
testing_number = int(0.2*count)

# Option 1 - keep it in chronological order
x_train = x[:-testing_number]
x_test = x[-testing_number:]
y_train = y[:-testing_number]
y_test = y[-testing_number:]

#ARIMA needs the time series to be in chronological order no matter what
y_arima_train = y_train
y_arima_test = y_test

# Option 2 - mix it completely
from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split (x,y,test_size = 0.2, random_state = 42)

# Setup machine learning algorithms (neural network, multiple linear regression, ARIMA & Random Walk)
# LR
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
y_pred_mlr = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mse)
r2_mlr = r2_score(y_test, y_pred_mlr)
print(f'Linear Regression\nRoot Mean Squared Error: {rmse_mlr}')
print(f'R-squared: {r2_mlr}')

# In case Neural network should not be calculated new, since it takes a long time
rmse_nn = 4.4
r2_nn = 0.997

# Neural Network
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter ("ignore", category = ConvergenceWarning)

param_list = {
    "hidden_layer_sizes": [(32,), (16,), (8,), (32, 16), (16, 8), (8, 4),(32,16,8,4), (16,8,4), (64, 32, 16, 8, 4)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["adam", "sgd"],
    "alpha": [0.00005, 0.0005]
}
'''
param_list = {"hidden_layer_sizes": [(1,), (50,)], 
              "activation": ["identity", "logistic", "tanh", "relu"], 
              "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}
'''
'''

param_list = {
    "hidden_layer_sizes": [(50,)],  # You can start with a single configuration
    "activation": ["relu"],  # Choose one activation function
    "solver": ["adam"],  # Choose one solver
    "alpha": [0.00005, 0.0005],
    "early_stopping": [True],  # Enable early stopping
    }
'''
clf = GridSearchCV(MLPRegressor(max_iter = 500),param_list, n_jobs=-1, cv=3)
#clf = MLPRegressor(hidden_layer_sizes=(64, 32, 16, 8, 4))
clf = clf.fit(x_train, y_train)

y_pred_nn = clf.predict(x_test)
mse = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse)
r2_nn = r2_score(y_test, y_pred_nn)
print(f'Neural Network \nRoot Mean Squared Error: {rmse_nn}')
print(f'R-squared: {r2_nn}')

 # Results - Root Mean Squared Error: 4.402212316017417 - R-squared: 0.997931640491701

# Random Walk / Alternative (Straight line, no changes)
value = 0
y_pred_rand = []
for i in range(len(y_arima_test)):
    y_pred_rand.append(value)
mse = mean_squared_error(y_arima_test,y_pred_rand)
rmse_rand = np.sqrt(mse)
r2_rand = r2_score(y_arima_test,y_pred_rand)
print(f'Random\nRoot Mean Squared Error: {rmse_rand}')
print(f'R-squared: {r2_rand}')

### Create 5 day moving average
days_average = 5
y_moving_average = y_test.rolling(window=days_average).mean()

### Momentum of stock price (using time length of time period of test time period)
length = len(x_test)
momentum = y_train[-length:]
value = momentum.mean()
momentum_list = []
for i in range(length):
    momentum_list.append(value)


### Visualize R-squared and RMSE per method

# Bring the forecasted results together
all_forecast = y_test.to_frame()

variables = [y_pred_mlr, y_pred_rand, y_pred_nn, y_moving_average, momentum_list] #add neural network
labels = ["Linear Regression", "Straight-Line", "Neural Network", "Actual Price (log_return) 5-day moving average ", "Momentum"] # add neural network

for i in labels:
    new_df = pd.DataFrame({i:variables[labels.index(i)]}) 
    new_df.reset_index(drop=True,inplace=True)
    all_forecast.reset_index(drop=True, inplace=True)
    all_forecast[i] = new_df[i]
    
all_forecast.index = y_test.index

#chart the prices
#import time-lag
import time

time.sleep(2.5)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(y_test,label="Actual price (log return)", marker = "o")
plt.title(company_ticker)
for i in labels:
    plt.plot(all_forecast[i], label = i, marker = "o")
plt.legend()
plt.show()



#r-squared
values = [r2_mlr, r2_nn, r2_rand]
labels = ["Linear Regression", "Neural Network", "Straight-line"]

plt.bar(labels,values)
plt.xlabel("Methods")
plt.ylabel("R-squared")
plt.title("R-Squared Results")
plt.show()

# Root-Mean-Squared-Error
values = [rmse_mlr, rmse_nn, rmse_rand]
labels = ["Linear Regression", "Neural Network", "Straight-line"]

plt.bar(labels,values)
plt.xlabel("Methods")
plt.ylabel("RMSE")
plt.title("Root-Mean-Squared-Error Results")
plt.show()

print(f'Neural Network \nRoot Mean Squared Error: {rmse_nn.round(2)}')
print(f'R-squared: {r2_nn.round(2)}')

print(f'Random\nRoot Mean Squared Error: {rmse_rand.round(2)}')
print(f'R-squared: {r2_rand.round(2)}')

print(f'Linear Regression\nRoot Mean Squared Error: {rmse_mlr.round(2)}')
print(f'R-squared: {r2_mlr.round(2)}')


