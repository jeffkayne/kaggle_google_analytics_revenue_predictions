from sklearn.preprocessing import LabelEncoder
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../google_analytics_customer_revenue_prediction/all (2)/"))


# Function to load csv and flatten JSON fields
def load_df(csv_path='../google_analytics_customer_revenue_prediction/all (2)/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column].tolist())
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# Load nrows of the data
train_df = load_df(nrows=1000)
test_df = load_df("../google_analytics_customer_revenue_prediction/all (2)/test.csv", nrows=100)

# combine train and test sets
merged_df = pd.concat([train_df, test_df])

# NaN values are not counted in obj.nunique() method, but may be meaningful in this dataset
unique_value_columns = [col for col in merged_df if merged_df[col].nunique() == 1]
columns_to_remove = []

# Drop columns with unique values
for col in unique_value_columns:
    if len(merged_df[col].unique()) > 1 and merged_df[col].unique()[1] != 'not available in demo dataset':
        continue
    columns_to_remove.append(col)
print('Removeable columns: ', columns_to_remove)
merged_df = merged_df.drop(columns_to_remove, axis=1)

# feature engineering on dates
format_str = '%Y%m%d'
merged_df['formated_date'] = merged_df['date'].apply(
    lambda x: datetime.strptime(str(x), format_str))
merged_df['month'] = merged_df['formated_date'].apply(lambda x: x.month)
merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x: x.day // 8)
merged_df['day'] = merged_df['formated_date'].apply(lambda x: x.day)
merged_df['weekday'] = merged_df['formated_date'].apply(lambda x: x.weekday())
del merged_df['date']
del merged_df['formated_date']

# Display what columns are still "unique"
unique_columns = []
for col in merged_df.columns:
    if len(merged_df[col].unique()) == 2:
        unique_columns.append(col)
        print('The column *', col, '* has unique values of: ', merged_df[col].unique())

# replace nan terms in relevant columns
merged_df['totals.bounces'] = merged_df['totals.bounces'].fillna('0')
merged_df['totals.newVisits'] = merged_df['totals.newVisits'].fillna('0')
merged_df['totals.transactionRevenue'] = merged_df['totals.transactionRevenue'].fillna('0')
merged_df['trafficSource.adwordsClickInfo.isVideoAd'] = merged_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(
    True)
merged_df['trafficSource.isTrueDirect'] = merged_df['trafficSource.isTrueDirect'].fillna(False)

# Divide data in to categorical and quantitative columns
cat_features = []
quant_features = []
for col in merged_df:
    if merged_df[col].dtype == object:
        cat_features.append(col)
        # print(col, ': ', train_dfbis[col].unique())
    elif merged_df[col].dtype == bool:
        merged_df.loc[:, col] = merged_df[col].astype(np.int64)
        quant_features.append(col)
    else:
        quant_features.append(col)
print('Categorical features: ', cat_features)
print('Quantitative features: ', quant_features)

# Not sure what visitId is exactly. Delete it for now.
# When you delete column, it is GONE! Think of better way to do this if you want to retrieve.
del merged_df['visitId']
# Extract hour and minute from visitStartTime
merged_df['visitStartTime'] = pd.to_datetime(merged_df['visitStartTime'], unit='s')
merged_df['hour'] = merged_df['visitStartTime'].apply(lambda x: x.hour)
merged_df['minute'] = merged_df['visitStartTime'].apply(lambda x: x.minute)
del merged_df['visitStartTime']

# Transform quantitative data to int (of float if too large)
merged_df.loc[:, 'totals.bounces'] = merged_df['totals.bounces'].astype(np.int64)
merged_df.loc[:, 'totals.hits'] = merged_df['totals.hits'].astype(np.int64)
merged_df.loc[:, 'totals.newVisits'] = merged_df['totals.newVisits'].astype(np.int64)
merged_df['trafficSource.adwordsClickInfo.page'] = merged_df['trafficSource.adwordsClickInfo.page'].fillna(
    0).astype(np.int64)
merged_df.loc[:, 'totals.pageviews'] = merged_df['totals.pageviews'].fillna(0).astype(np.int64)
merged_df.loc[:, 'totals.transactionRevenue'] = merged_df['totals.transactionRevenue'].fillna(
    0).astype(np.int64)
merged_df.loc[:, 'fullVisitorId'] = merged_df['fullVisitorId'].astype(np.float_)

# Encode all categorical data
cat_features = merged_df[cat_features].select_dtypes(include=['object']).columns
lb_make = LabelEncoder()
ohe_cols = []
for col in cat_features:
    if col == 'fullVisitorId':
        continue
    if merged_df[col].nunique() < 50:
        ohe_cols.append(col)
        continue
    merged_df[col] = lb_make.fit_transform(merged_df[col].astype(str))
merged_df = pd.get_dummies(merged_df, columns=ohe_cols)

# SLice merge_df back into train and test sets
train_clean = merged_df[:len(train_df)]
test_clean = merged_df[len(train_df):]
del test_clean['totals.transactionRevenue']

# validate predictions on training set (spliting it to train and test sets)
y = train_clean['totals.transactionRevenue'].apply(lambda a: np.log1p(a) if a > 0 else 0)
X = train_clean.loc[:, train_clean.columns != 'totals.transactionRevenue']

# transaction level predictions
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state=1)
# take out fullVisitorId for prediction
val_X_bis = val_X.loc[:, train_X.columns != 'fullVisitorId']
train_X_bis = train_X.loc[:, train_X.columns != 'fullVisitorId']

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X_bis, train_y)
rf_val_predictions = rf_model.predict(val_X_bis)
rf_val_mae = np.sqrt(mean_squared_error(val_y, rf_val_predictions))
print("Validation MAE for Random Forest Model: ", rf_val_mae)

# concat rf_val_predictions back on to val_X df
y_predict = pd.Series(rf_val_predictions, name='predicted.transactionRevenue', index=val_X.index)
visitor_revenue = pd.concat([y_predict, val_y, val_X], axis=1).groupby('fullVisitorId')['predicted.transactionRevenue', 'totals.transactionRevenue'].sum()
result = np.sqrt(mean_squared_error(visitor_revenue['predicted.transactionRevenue'], visitor_revenue['totals.transactionRevenue']))
print('Result = ', result)
