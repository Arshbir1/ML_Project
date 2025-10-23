"""
TRAIN_OVER.PY - Moderate Complexity Increase from train_optimized
Target: Capture more patterns while controlling overfitting
Changes: +8 features, +1 base model, +100 trees, tuned lr
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                             ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import sys

print("="*100)
print(" " * 20 + "TRAIN_OVER - MODERATE COMPLEXITY INCREASE")
print(" " * 15 + "From train_optimized (24,289) → More Features + Model")
print("="*100)

# ============================
# PHASE 1: DATA LOADING
# ============================
print("\n" + "="*100)
print("PHASE 1: DATA LOADING")
print("="*100)

try:
    df = pd.read_csv("train.csv")
    print(f"\n✓ train.csv loaded: {df.shape}")
except FileNotFoundError:
    print("\n✗ Error: train.csv not found.")
    sys.exit(1)

X = df.drop(['HotelValue', 'Id'], axis=1).copy()
y = df['HotelValue'].copy()
y_log = np.log1p(y)

print(f"  Target skewness: {y.skew():.3f}")

# ============================
# PHASE 2: ENHANCED FEATURE ENGINEERING
# ============================
print("\n" + "="*100)
print("PHASE 2: ENHANCED FEATURE ENGINEERING (train_optimized + 8 new features)")
print("="*100)

print(f"\nOriginal features: {X.shape[1]}")

# --- Core features from train_optimized ---
# Age features
X['PropertyAge'] = X['YearSold'] - X['ConstructionYear']
X['RenovationYear'] = X['RenovationYear'].fillna(X['ConstructionYear'])
X['YearsSinceRenovation'] = X['YearSold'] - X['RenovationYear']
X['ParkingConstructionYear'] = X['ParkingConstructionYear'].fillna(X['YearSold'])
X['ParkingAge'] = X['YearSold'] - X['ParkingConstructionYear']

# Area features
X['TotalLivingArea'] = X['GroundFloorArea'] + X['UpperFloorArea']
X['TotalPropertyArea'] = X['TotalLivingArea'] + X['BasementTotalSF']
X['TotalOutdoorArea'] = (X['TerraceArea'] + X['OpenVerandaArea'] + 
                         X['EnclosedVerandaArea'] + X['SeasonalPorchArea'] + 
                         X['ScreenPorchArea'])
X['BasementRatio'] = X['BasementTotalSF'] / (X['TotalPropertyArea'] + 1)
X['ParkingToLandRatio'] = X['ParkingArea'] / (X['LandArea'] + 1)

# Quality interactions
X['QualityAreaInteraction'] = X['OverallQuality'] * X['UsableArea']
X['QualityRoomsInteraction'] = X['OverallQuality'] * X['GuestRooms']
X['QualityConditionInteraction'] = X['OverallQuality'] * X['OverallCondition']

# Bathroom features
X['TotalBathrooms'] = (X['FullBaths'] + X['HalfBaths'] * 0.5 + 
                       X['BasementFullBaths'] + X['BasementHalfBaths'] * 0.5)
X['BathroomToRoomRatio'] = X['TotalBathrooms'] / (X['GuestRooms'] + 1)
X['RoomsPerArea'] = X['GuestRooms'] / (X['UsableArea'] + 1)

# Amenity scores
X['HasPool'] = (X['SwimmingPoolArea'] > 0).astype(int)
X['HasParking'] = (X['ParkingArea'] > 0).astype(int)
X['HasBasement'] = (X['BasementTotalSF'] > 0).astype(int)
X['AmenityScore'] = X['HasPool'] + X['HasParking'] + X['HasBasement'] + X['Lounges']

# Renovation
X['IsRenovated'] = (X['RenovationYear'] != X['ConstructionYear']).astype(int)
X['RenovationImpact'] = X['IsRenovated'] * X['YearsSinceRenovation']

# --- NEW FEATURES (8 additional) ---
print("\nAdding 8 new features for increased complexity...")

# 1-2: More quality interactions
X['QualityParkingInteraction'] = X['OverallQuality'] * X['ParkingCapacity']
X['QualityBasementInteraction'] = X['OverallQuality'] * X['BasementTotalSF']

# 3-4: Spatial efficiency features
X['LandAreaPerRoom'] = X['LandArea'] / (X['GuestRooms'] + 1)
X['ParkingEfficiency'] = X['ParkingCapacity'] / (X['ParkingArea'] + 1)

# 5: Premium property indicator
X['IsPremium'] = ((X['OverallQuality'] >= 8) & (X['UsableArea'] > X['UsableArea'].median())).astype(int)

# 6: Basement facility features
X['TotalBasementFacilitySF'] = X['BasementFacilitySF1'].fillna(0) + X['BasementFacilitySF2'].fillna(0)
X['FinishedBasementRatio'] = X['TotalBasementFacilitySF'] / (X['BasementTotalSF'] + 1)

# 7: Outdoor space ratio
X['OutdoorToTotalRatio'] = X['TotalOutdoorArea'] / (X['TotalPropertyArea'] + 1)

# 8: Age-quality interaction
X['AgeQualityInteraction'] = X['PropertyAge'] * X['OverallQuality']

# Drop temporal columns
drop_cols = ['ConstructionYear', 'RenovationYear', 'ParkingConstructionYear', 'YearSold', 'MonthSold']
X = X.drop(drop_cols, axis=1)

print(f"✓ Total features after engineering: {X.shape[1]}")
print(f"  New features added: 8")
print(f"  Total engineered: ~28 features")

# ============================
# PHASE 3: PREPROCESSING
# ============================
print("\n" + "="*100)
print("PHASE 3: PREPROCESSING PIPELINE")
print("="*100)

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

ordinal_features = [
    'ExteriorQuality', 'ExteriorCondition', 'BasementHeight', 'BasementCondition',
    'BasementExposure', 'HeatingQuality', 'CentralAC', 'KitchenQuality',
    'PropertyFunctionality', 'LoungeQuality', 'ParkingFinish', 'ParkingQuality',
    'ParkingCondition', 'DrivewayType', 'PoolQuality', 'BoundaryFence',
    'PlotShape', 'LandSlope', 'UtilityAccess', 'LandElevation'
]
ordinal_features = [col for col in ordinal_features if col in X.columns]
ohe_features = [col for col in categorical_features if col not in ordinal_features]

quality_scale = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
bsmt_exp_scale = ['None', 'No', 'Mn', 'Av', 'Gd']
bsmt_hgt_scale = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
func_scale = ['None', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
park_fin_scale = ['None', 'Unf', 'RFn', 'Fin']
drive_scale = ['None', 'N', 'P', 'Y']
fence_scale = ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
shape_scale = ['None', 'IR3', 'IR2', 'IR1', 'Reg']
slope_scale = ['None', 'Sev', 'Mod', 'Gtl']
util_scale = ['None', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub']
elev_scale = ['None', 'Low', 'HLS', 'Bnk', 'Lvl']

ordinal_categories = [
    quality_scale, quality_scale, bsmt_hgt_scale, quality_scale,
    bsmt_exp_scale, quality_scale, ['N', 'Y'], quality_scale,
    func_scale, quality_scale, park_fin_scale, quality_scale,
    quality_scale, drive_scale, quality_scale, fence_scale,
    shape_scale, slope_scale, util_scale, elev_scale
]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories,
                              handle_unknown='use_encoded_value',
                              unknown_value=-1))
])

ohe_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('ohe', ohe_transformer, ohe_features)
    ],
    remainder='drop'
)

print(f"\n✓ Preprocessing configured")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Ordinal: {len(ordinal_features)}")
print(f"  One-hot: {len(ohe_features)}")

# ============================
# PHASE 4: ENHANCED STACKING ENSEMBLE
# ============================
print("\n" + "="*100)
print("PHASE 4: ENHANCED 5-MODEL STACKING ENSEMBLE")
print("="*100)

print("\nBase learners (Layer 1):")
print("  1. GradientBoosting: n_estimators=600 (↑), lr=0.045 (↓), depth=4")
print("  2. RandomForest: n_estimators=300, depth=15")
print("  3. ExtraTrees: n_estimators=200, depth=15")
print("  4. Ridge: alpha=10.0")
print("  5. AdaBoost: n_estimators=100 (NEW)")

layer_1_estimators = [
    ('gb', GradientBoostingRegressor(n_estimators=600, learning_rate=0.045,
                                    max_depth=4, min_samples_split=20,
                                    subsample=0.8, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, 
                                min_samples_split=10, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('ridge', Ridge(alpha=10.0, random_state=42)),
    ('ada', AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
]

meta_learner = Ridge(alpha=5.0, random_state=42)

stacking_regressor = StackingRegressor(
    estimators=layer_1_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking_regressor)
])

print("\nMeta-learner: Ridge(alpha=5.0)")
print("\n⚠️  Warning: Increased complexity - monitor for overfitting")

# ============================
# PHASE 5: TRAINING
# ============================
print("\n" + "="*100)
print("PHASE 5: TRAINING FINAL MODEL")
print("="*100)

print("\nTraining 5-model stacking ensemble on full dataset...")
print("(This may take 5-10 minutes due to increased complexity)")

final_pipeline.fit(X, y_log)
print("\n✓ Training complete")

# ============================
# PHASE 6: PREDICTION
# ============================
print("\n" + "="*100)
print("PHASE 6: TEST PREDICTION")
print("="*100)

try:
    df_test = pd.read_csv("test.csv")
    print(f"\n✓ test.csv loaded: {df_test.shape}")

    test_ids = df_test['Id']
    X_test = df_test.drop('Id', axis=1).copy()

    # Apply SAME feature engineering
    print("\nApplying feature engineering...")

    # Core features
    X_test['PropertyAge'] = X_test['YearSold'] - X_test['ConstructionYear']
    X_test['RenovationYear'] = X_test['RenovationYear'].fillna(X_test['ConstructionYear'])
    X_test['YearsSinceRenovation'] = X_test['YearSold'] - X_test['RenovationYear']
    X_test['ParkingConstructionYear'] = X_test['ParkingConstructionYear'].fillna(X_test['YearSold'])
    X_test['ParkingAge'] = X_test['YearSold'] - X_test['ParkingConstructionYear']

    X_test['TotalLivingArea'] = X_test['GroundFloorArea'] + X_test['UpperFloorArea']
    X_test['TotalPropertyArea'] = X_test['TotalLivingArea'] + X_test['BasementTotalSF']
    X_test['TotalOutdoorArea'] = (X_test['TerraceArea'] + X_test['OpenVerandaArea'] + 
                                  X_test['EnclosedVerandaArea'] + X_test['SeasonalPorchArea'] + 
                                  X_test['ScreenPorchArea'])
    X_test['BasementRatio'] = X_test['BasementTotalSF'] / (X_test['TotalPropertyArea'] + 1)
    X_test['ParkingToLandRatio'] = X_test['ParkingArea'] / (X_test['LandArea'] + 1)

    X_test['QualityAreaInteraction'] = X_test['OverallQuality'] * X_test['UsableArea']
    X_test['QualityRoomsInteraction'] = X_test['OverallQuality'] * X_test['GuestRooms']
    X_test['QualityConditionInteraction'] = X_test['OverallQuality'] * X_test['OverallCondition']

    X_test['TotalBathrooms'] = (X_test['FullBaths'] + X_test['HalfBaths'] * 0.5 + 
                                X_test['BasementFullBaths'] + X_test['BasementHalfBaths'] * 0.5)
    X_test['BathroomToRoomRatio'] = X_test['TotalBathrooms'] / (X_test['GuestRooms'] + 1)
    X_test['RoomsPerArea'] = X_test['GuestRooms'] / (X_test['UsableArea'] + 1)

    X_test['HasPool'] = (X_test['SwimmingPoolArea'] > 0).astype(int)
    X_test['HasParking'] = (X_test['ParkingArea'] > 0).astype(int)
    X_test['HasBasement'] = (X_test['BasementTotalSF'] > 0).astype(int)
    X_test['AmenityScore'] = X_test['HasPool'] + X_test['HasParking'] + X_test['HasBasement'] + X_test['Lounges']

    X_test['IsRenovated'] = (X_test['RenovationYear'] != X_test['ConstructionYear']).astype(int)
    X_test['RenovationImpact'] = X_test['IsRenovated'] * X_test['YearsSinceRenovation']

    # NEW features
    X_test['QualityParkingInteraction'] = X_test['OverallQuality'] * X_test['ParkingCapacity']
    X_test['QualityBasementInteraction'] = X_test['OverallQuality'] * X_test['BasementTotalSF']
    X_test['LandAreaPerRoom'] = X_test['LandArea'] / (X_test['GuestRooms'] + 1)
    X_test['ParkingEfficiency'] = X_test['ParkingCapacity'] / (X_test['ParkingArea'] + 1)
    X_test['IsPremium'] = ((X_test['OverallQuality'] >= 8) & (X_test['UsableArea'] > X_test['UsableArea'].median())).astype(int)
    X_test['TotalBasementFacilitySF'] = X_test['BasementFacilitySF1'].fillna(0) + X_test['BasementFacilitySF2'].fillna(0)
    X_test['FinishedBasementRatio'] = X_test['TotalBasementFacilitySF'] / (X_test['BasementTotalSF'] + 1)
    X_test['OutdoorToTotalRatio'] = X_test['TotalOutdoorArea'] / (X_test['TotalPropertyArea'] + 1)
    X_test['AgeQualityInteraction'] = X_test['PropertyAge'] * X_test['OverallQuality']

    X_test = X_test.drop(drop_cols, axis=1)
    X_test = X_test[X.columns]

    print("✓ Feature engineering applied")

    # Predict
    y_test_log_pred = final_pipeline.predict(X_test)
    y_test_pred = np.expm1(y_test_log_pred)
    y_test_pred = np.clip(y_test_pred, 0, None)

    submission = pd.DataFrame({
        "Id": test_ids,
        "HotelValue": y_test_pred
    })

    submission.to_csv("submission.csv", index=False)

    print("\n" + "="*100)
    print("✓✓✓ SUBMISSION.CSV CREATED - TRAIN_OVER (Increased Complexity)")
    print("="*100)
    print(f"\nChanges from train_optimized:")
    print(f"  + 8 new features (28 total)")
    print(f"  + 1 additional base model (AdaBoost)")
    print(f"  + 100 more GB trees (600 total)")
    print(f"  - Slightly lower learning rate (0.045)")
    print(f"\nExpected: Slightly better if patterns captured, or slight overfit")

except FileNotFoundError:
    print("\n⚠ test.csv not found")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*100)
print("TRAINING COMPLETE - MODERATE COMPLEXITY INCREASE")
print("="*100)
