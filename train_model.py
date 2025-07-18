import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv('Employers_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['Employee_ID', 'Name'])

# Features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Categorical & numerical columns
categorical_cols = ['Gender', 'Department', 'Job_Title', 'Education_Level', 'Location']
numerical_cols = ['Age', 'Experience_Years']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Full pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model trained successfully. MSE: {mse:.2f}")

# Save model
joblib.dump(model, 'salary_model.pkl')
