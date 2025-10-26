import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
# --- Models ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

# --- Metrics ---
from sklearn.metrics import mean_squared_error

# Load the dataset
try:
    df = pd.read_csv("train.csv")
    print("train.csv loaded successfully.")
    print(f"Shape of the data: {df.shape}\n")
    
    # 1. Separate features (X) and target (y)
    # We drop HotelValue and Id from features
    X = df.drop(['HotelValue', 'Id'], axis=1)
    
    # We will log-transform the target variable. This is a crucial step for
    # house price prediction as it normalizes the skewed distribution.
    # We use np.log1p which is log(1 + x) to handle any potential 0 values.
    y = np.log1p(df['HotelValue'])
    
    print("Separated features (X) and log-transformed target (y).\n")
    print("Original 'HotelValue' head:")
    print(df['HotelValue'].head())
    print("\nLog-transformed 'y' head:")
    print(y.head())
    
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure it's in the correct directory.")
    sys.exit(1)
# We'll work with the 'X' DataFrame created in the previous step.

print("--- Phase 2: Feature Engineering (Creating 'Age' Features) ---")

# 1. Create 'PropertyAge'
X['PropertyAge'] = X['YearSold'] - X['ConstructionYear']

# 2. Create 'YearsSinceRenovation'
# If RenovationYear is NaN, it means it was never renovated.
# We can set it to be the same as ConstructionYear.
X['RenovationYear'] = X['RenovationYear'].fillna(X['ConstructionYear'])
X['YearsSinceRenovation'] = X['YearSold'] - X['RenovationYear']

# 3. Create 'ParkingAge'
# If ParkingConstructionYear is NaN, it might mean no parking.
# We'll fillna with YearSold to make ParkingAge = 0, or with a distinct value.
# A safe bet is to fill with the median parking construction year, or just fill with 0
# For this, let's fill with YearSold, so age is 0 if NaN.
X['ParkingConstructionYear'] = X['ParkingConstructionYear'].fillna(X['YearSold'])
X['ParkingAge'] = X['YearSold'] - X['ParkingConstructionYear']

# 4. Drop the original year columns
# We also drop MonthSold as it's less likely to be as useful as the engineered age features.
X = X.drop(['ConstructionYear', 'RenovationYear', 'ParkingConstructionYear', 'YearSold', 'MonthSold'], axis=1)

print("Engineered features created:")
print(X[['PropertyAge', 'YearsSinceRenovation', 'ParkingAge']].head())
print(f"\nNew shape of X: {X.shape}")


print("\n--- Phase 3: Identifying Column Types for Preprocessing ---")

# 1. Identify Numerical Features
# These are all columns with a number dtype.
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
print(f"Found {len(numerical_features)} numerical features.")

# 2. Identify Ordinal Features (Manual List)
# These are categorical features with a clear order.
ordinal_features = [
    'ExteriorQuality', 'ExteriorCondition', 'BasementHeight', 'BasementCondition',
    'BasementExposure', 'HeatingQuality', 'CentralAC', 'KitchenQuality', 
    'PropertyFunctionality', 'LoungeQuality', 'ParkingFinish', 'ParkingQuality', 
    'ParkingCondition', 'DrivewayType', 'PoolQuality', 'BoundaryFence',
    'PlotShape', 'LandSlope', 'UtilityAccess', 'LandElevation'
]
# Filter list to only include columns that are actually in X
ordinal_features = [col for col in ordinal_features if col in X.columns]
print(f"Found {len(ordinal_features)} ordinal features.")

# 3. Identify One-Hot Encoded (OHE) Features
# These are all 'object' type columns that are NOT in our ordinal list.
categorical_features = X.select_dtypes(include='object').columns.tolist()
ohe_features = [col for col in categorical_features if col not in ordinal_features]
print(f"Found {len(ohe_features)} OHE features.")


# --- Define the specific order for our Ordinal Features ---
# This is crucial for the OrdinalEncoder.
# We'll create a list of categories. Note: 'None' is added for missing values.

# Quality scale: Ex, Gd, TA, Fa, Po
quality_scale = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
# Basement Exposure scale
bsmt_exp_scale = ['None', 'No', 'Mn', 'Av', 'Gd']
# Basement Height scale
bsmt_hgt_scale = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'] # Assuming 'Po' for poor/low
# Functionality scale
func_scale = ['None', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
# Parking Finish scale
park_fin_scale = ['None', 'Unf', 'RFn', 'Fin']
# Driveway scale
drive_scale = ['None', 'N', 'P', 'Y']
# Fence scale
fence_scale = ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
# Plot Shape
shape_scale = ['None', 'IR3', 'IR2', 'IR1', 'Reg']
# Land Slope
slope_scale = ['None', 'Sev', 'Mod', 'Gtl']
# Utility Access
util_scale = ['None', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub']
# Land Elevation
elev_scale = ['None', 'Low', 'HLS', 'Bnk', 'Lvl']

# List of all category lists.
# IMPORTANT: This list MUST be in the same order as `ordinal_features`
ordinal_categories = [
    quality_scale, # ExteriorQuality
    quality_scale, # ExteriorCondition
    bsmt_hgt_scale, # BasementHeight
    quality_scale, # BasementCondition
    bsmt_exp_scale, # BasementExposure
    quality_scale, # HeatingQuality
    ['N', 'Y'], # CentralAC
    quality_scale, # KitchenQuality
    func_scale, # PropertyFunctionality
    quality_scale, # LoungeQuality
    park_fin_scale, # ParkingFinish
    quality_scale, # ParkingQuality
    quality_scale, # ParkingCondition
    drive_scale, # DrivewayType
    quality_scale, # PoolQuality
    fence_scale, # BoundaryFence
    shape_scale, # PlotShape
    slope_scale, # LandSlope
    util_scale, # UtilityAccess
    elev_scale  # LandElevation
]

# We need to split our data *before* fitting the preprocessor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training/test sets. X_train shape: {X_train.shape}")

print("\n--- Phase 4: Building Preprocessing Pipeline & Training All Models ---")

# 1. Define the 3 processing pipelines
# We use the feature lists created in the previous step.

# --- Numerical Pipeline ---
# Impute missing values (e.g., RoadAccessLength) with the median
# Scale all features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Ordinal Pipeline ---
# Impute missing values with the string "None"
# Encode using the specific categories we defined
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories, 
                               handle_unknown='use_encoded_value', 
                               unknown_value=-1)) # Use -1 for unknown categories
])

# --- One-Hot Encoding Pipeline ---
# Impute missing values with the string "None"
# Create new binary columns for each category
ohe_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# 2. Combine pipelines in ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('ohe', ohe_transformer, ohe_features)
    ],
    remainder='passthrough' # Keep any columns we didn't specify
)

# 3. Define all the models for our experiment
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.001, random_state=42), # Small alpha for baseline
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "BaggingRegressor": BaggingRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
}

# 4. Loop, build full pipeline, cross-validate, and store results
results = {}

print("Starting model training and cross-validation...")
print("This may take a few minutes...")

for name, model in models.items():
    
    # Create the full pipeline: Preprocess -> Train
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Evaluate the pipeline using 5-fold cross-validation
    # We use 'X_train' and 'y_train' (the 80% split)
    scores = cross_val_score(model_pipeline, 
                             X_train, 
                             y_train, 
                             cv=5, 
                             scoring='neg_root_mean_squared_error')
    
    # Store the mean and std dev of the scores
    results[name] = {
        'mean_rmsle': -np.mean(scores), # We negate the score to make it positive
        'std_rmsle': np.std(scores)
    }
    print(f"  ... {name} complete.")

print("\n--- Model Comparison (RMSLE) ---")
print("Lower is better.")

# Convert results to a DataFrame for easy sorting and viewing
results_df = pd.DataFrame(results).T.sort_values(by='mean_rmsle')
results_df['mean_rmsle'] = results_df['mean_rmsle'].round(5)
results_df['std_rmsle'] = results_df['std_rmsle'].round(5)

print(results_df)

print("\n--- Phase 5: Hyperparameter Tuning (Example on GradientBoostingRegressor) ---")

# 1. Create the final pipeline object
# We'll tune the 'model' part of this pipeline
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# 2. Define a "parameter grid" to search
# We tell GridSearchCV which parameters to try on the 'model' step
# 'model__n_estimators' means "the 'n_estimators' parameter of the 'model' step"
param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1]
}

# 3. Set up the GridSearchCV
# cv=3 (3-fold cross-validation) is faster for a grid search
# n_jobs=-1 uses all your CPU cores to speed it up
grid_search = GridSearchCV(gb_pipeline, 
                           param_grid, 
                           cv=3, 
                           scoring='neg_root_mean_squared_error', 
                           n_jobs=-1,
                           verbose=1) # verbose=1 shows progress

print("Starting GridSearchCV... This will take a few minutes.")
# We fit on the 80% training set
grid_search.fit(X_train, y_train)

print("\nGridSearchCV complete.")
print(f"Best RMSLE score: {-grid_search.best_score_:.5f}")
print("Best parameters found:")
print(grid_search.best_params_)

# You can also test the best model on your 20% hold-out test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)
test_rmsle = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f"\nRMSLE on hold-out test set: {test_rmsle:.5f}")

print("\n--- Phase 6: Final Model Training and Submission ---")

# 1. Define the final, best model using the parameters from GridSearch
# Best params: {'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__n_estimators': 500}
final_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# 2. Create the final pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)
])

# 3. Train the final pipeline on ALL training data
# We use the full 'X' and 'y' from Phase 1
print("Training final model on all 1460 samples...")
final_pipeline.fit(X, y) # X and y are the full datasets
print("Final model training complete.")


# 4. Load the test.csv data
try:
    df_test = pd.read_csv("test.csv")
    print("\nLoaded test.csv successfully.")
    
    # Keep the Id column for the submission file
    test_ids = df_test['Id']
    
    # 5. Apply the EXACT same feature engineering to test.csv
    # Use 'X_test' as the variable name for the features
    X_test_final = df_test.drop('Id', axis=1)
    
    X_test_final['PropertyAge'] = X_test_final['YearSold'] - X_test_final['ConstructionYear']
    
    X_test_final['RenovationYear'] = X_test_final['RenovationYear'].fillna(X_test_final['ConstructionYear'])
    X_test_final['YearsSinceRenovation'] = X_test_final['YearSold'] - X_test_final['RenovationYear']
    
    X_test_final['ParkingConstructionYear'] = X_test_final['ParkingConstructionYear'].fillna(X_test_final['YearSold'])
    X_test_final['ParkingAge'] = X_test_final['YearSold'] - X_test_final['ParkingConstructionYear']
    
    # Drop the original year columns
    X_test_final = X_test_final.drop(['ConstructionYear', 'RenovationYear', 'ParkingConstructionYear', 'YearSold', 'MonthSold'], axis=1)
    
    # Ensure X_test_final has the same columns in the same order as X (the training features)
    # This handles any case where the test set might be missing a category, etc.
    X_test_final = X_test_final[X.columns] 

    print("Applied feature engineering to test.csv.")

    # 6. Make predictions
    # The pipeline will automatically apply all preprocessing
    print("Making predictions on test set...")
    final_log_predictions = final_pipeline.predict(X_test_final)
    
    # 7. Reverse the log-transform
    # We use np.expm1() which is the inverse of np.log1p()
    final_predictions = np.expm1(final_log_predictions)
    
    # 8. Create the submission file
    submission = pd.DataFrame({
        "Id": test_ids,
        "HotelValue": final_predictions
    })
    
    # Ensure no negative predictions (just in case)
    submission['HotelValue'] = submission['HotelValue'].clip(lower=0)
    
    submission.to_csv("submission.csv", index=False)
    
    print("\nSubmission file created successfully!")
    print(submission.head())

except FileNotFoundError:
    print("\n--- test.csv not found. ---")
    print("Once you have your 'test.csv' file from Kaggle,")
    print("you can run the code in this block to generate your submission.")

except KeyError as e:
    print(f"\n--- Error during test set processing ---")
    print(f"Column not found: {e}. This likely means 'test.csv' has a different structure than 'train.csv'.")
    print("Ensure all feature engineering steps are applied correctly.")
