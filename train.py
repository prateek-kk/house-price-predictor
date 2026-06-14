import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, root_mean_squared_error )

# Load the dataset
train_df= pd.read_csv('data/train.csv')
test_df= pd.read_csv('data/test.csv')

print(train_df.shape)
print(test_df.shape)

print(train_df.columns)
y= train_df['SalePrice']
x= train_df.drop(['Id', 'SalePrice'], axis=1)
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.dtypes.value_counts())

print(x.isnull().sum().sort_values(ascending=False).head(15))

x= x.drop([ "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence"], axis=1)


# Fill numeric missing values with median
x["LotFrontage"] = x["LotFrontage"].fillna(
    x["LotFrontage"].median()
)

# Fill remaining numeric missing values with median
numeric_cols = x.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_cols:
    x[col] = x[col].fillna(x[col].median())


# Fill categorical missing values with mode
categorical_cols = x.select_dtypes(
    include=["str"]
).columns

for col in categorical_cols:
    x[col] = x[col].fillna(x[col].mode()[0])

# One-hot encoding
x = pd.get_dummies(x)

print("Final Shape:", x.shape)
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

preds = model.predict(x_test)

mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)
print(x.shape)

rmse = root_mean_squared_error(y_test, preds)

print("RMSE:", rmse)

print(y.min())
print(y.max())
print(y.mean())

scores = -cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print(scores)
print(scores.mean())
