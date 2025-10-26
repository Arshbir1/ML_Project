# hotel_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import warnings

warnings.filterwarnings('ignore')

# --- LOAD DATA ---
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"{e}")
    exit()

# --- ID column detection ---
id_col = 'Id' if 'Id' in train.columns else 'id'
print(f" Using '{id_col}' as ID column")

train_ids = train[id_col]
test_ids = test[id_col]
train = train.drop(id_col, axis=1)
test = test.drop(id_col, axis=1)

# --- TARGET COLUMN ---
target_col = 'HotelValue'
if target_col not in train.columns:
    raise ValueError(" 'HotelValue' column not found in train.csv!")

# --- FIX NEGATIVE VALUES ---
# Any negative numeric values that don’t make sense are replaced with 0
num_cols = train.select_dtypes(include=[np.number]).columns
for col in num_cols:
    neg_count = (train[col] < 0).sum()
    if neg_count > 0:
        print(f" Found {neg_count} negative values in '{col}', replacing with 0")
        train[col] = np.where(train[col] < 0, 0, train[col])

# --- DELIVERY / ORDER FIX ---
# If both columns exist, fix delivery < order problem
if 'DeliveryDay' in train.columns and 'OrderDay' in train.columns:
    invalid_rows = train[train['DeliveryDay'] < train['OrderDay']].shape[0]
    print(f" Found {invalid_rows} rows where DeliveryDay < OrderDay")
    train.loc[train['DeliveryDay'] < train['OrderDay'], ['DeliveryDay', 'OrderDay']] = np.nan

# --- DROP TARGET ---
y = np.log1p(train[target_col])  # log-transform target
train = train.drop(columns=[target_col])

# --- COMBINE FOR CLEAN PREPROCESSING ---
all_data = pd.concat([train, test], ignore_index=True)

# --- HANDLE MISSING VALUES ---
for col in ['PoolQuality', 'ExtraFacility', 'FacadeType', 'BoundaryFence', 'LoungeQuality',
            'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition',
            'BasementHeight', 'BasementCondition', 'BasementExposure',
            'BasementFacilityType1', 'BasementFacilityType2']:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

for col in all_data.columns:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# --- FEATURE ENGINEERING ---
if 'YearSold' in all_data.columns and 'ConstructionYear' in all_data.columns:
    all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
if 'RenovationYear' in all_data.columns and 'YearSold' in all_data.columns:
    all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
if all(x in all_data.columns for x in ['BasementTotalSF', 'GroundFloorArea', 'UpperFloorArea']):
    all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
if all(x in all_data.columns for x in ['FullBaths', 'HalfBaths', 'BasementFullBaths', 'BasementHalfBaths']):
    all_data['TotalBathrooms'] = (
        all_data['FullBaths'] +
        0.5 * all_data['HalfBaths'] +
        all_data['BasementFullBaths'] +
        0.5 * all_data['BasementHalfBaths']
    )

# --- ENCODING ---
all_data = pd.get_dummies(all_data, drop_first=True)

# --- SPLIT BACK ---
X = all_data[:len(train)]
X_test = all_data[len(train):]

# --- SCALING ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- RIDGE & LASSO MODELS ---
alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1]
alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

best_model, best_score, best_alpha = None, -999, None

print("\n--- Cross-Validation Results ---")
for alpha in alphas_ridge:
    model = Ridge(alpha=alpha)
    score = np.mean(cross_val_score(model, X_scaled, y, cv=10, scoring='r2'))
    print(f"Ridge (α={alpha}): {score:.4f}")
    if score > best_score:
        best_model, best_score, best_alpha = ('ridge', score, alpha)

for alpha in alphas_lasso:
    model = Lasso(alpha=alpha, max_iter=10000)
    score = np.mean(cross_val_score(model, X_scaled, y, cv=10, scoring='r2'))
    print(f"Lasso (α={alpha}): {score:.4f}")
    if score > best_score:
        best_model, best_score, best_alpha = ('lasso', score, alpha)

print(f"\n Best Model: {best_model.upper()} (α={best_alpha}) with CV R² = {best_score:.4f}")

# --- FINAL MODEL TRAINING ---
if best_model == 'ridge':
    final_model = Ridge(alpha=best_alpha)
else:
    final_model = Lasso(alpha=best_alpha, max_iter=10000)

final_model.fit(X_scaled, y)
preds_log = final_model.predict(X_test_scaled)
preds = np.expm1(preds_log)

submission = pd.DataFrame({id_col: test_ids, target_col: preds})
submission.to_csv('submission.csv', index=False)

print("\n 'submission.csv' created successfully!")


