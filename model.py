import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectFromModel
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the data
def load_data(train_path='train.csv', test_path='test.csv'):
    """
    Load train and test datasets with error handling
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Advanced Feature Engineering
def engineer_features(df):
    """
    Create advanced features from existing data
    """
    if df is None:
        return None
    
    # Convert date columns
    df['trans_date'] = pd.to_datetime(df['trans_date'])
    
    # Calculate age if 'dob' exists
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (datetime.now().year - df['dob'].dt.year)
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], 
            bins=[0, 18, 25, 35, 45, 55, 65, 100], 
            labels=['Under 18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
    
    # Extract time-based features
    df['hour_of_day'] = df['trans_date'].dt.hour
    df['day_of_week'] = df['trans_date'].dt.dayofweek
    df['month'] = df['trans_date'].dt.month
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate distance between cardholder and merchant
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Radius of earth in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    # Add merchant distance
    df['merchant_distance'] = haversine_distance(
        df['lat'], df['long'], 
        df['merch_lat'], df['merch_long']
    )
    
    # Transaction amount features
    df['amt_log'] = np.log1p(df['amt'])
    df['amt_zscore'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()
    
    # Categorical interaction features
    df['city_state_combo'] = df['city'] + '_' + df['state']
    
    return df

# Preprocessing and Model Training
def create_preprocessing_pipeline(feature_cols, categorical_features):
    """
    Create advanced preprocessing pipeline for features
    """
    # Preprocessing for numerical features
    numeric_features = [
        col for col in feature_cols if col not in categorical_features and col != 'is_fraud'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_model(X, y):
    """
    Train multiple models and ensemble them
    """
    # Create preprocessing pipeline
    categorical_features = [
        'category', 'gender', 'state', 'job', 
        'city_state_combo', 'age_group'
    ]
    preprocessor = create_preprocessing_pipeline(X.columns, categorical_features)
    
    # Hyperparameter grid for RandomForestClassifier
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Create full pipeline with model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print("Cross-validation ROC AUC scores:", cv_scores)
    print("Mean CV ROC AUC: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std() * 2))
    
    # Fit the model on full training data
    model.fit(X, y)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Detailed model evaluation
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")

def main():
    # Load data
    train_df, test_df = load_data()
    
    if train_df is None or test_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Engineer features for training data
    train_df = engineer_features(train_df)
    
    # Prepare feature columns
    feature_cols = [
        'amt_log', 'merchant_distance', 
        'hour_of_day', 'day_of_week', 'city_pop',
        'category', 'gender', 'state', 'job', 
        'age', 'age_group', 'is_weekend', 'amt_zscore',
        'city_state_combo', 'month'
    ]
    
    # Split features and target
    X = train_df[feature_cols]
    y = train_df['is_fraud']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate on validation set
    evaluate_model(model, X_val, y_val)
    
    # Engineer features for test data
    test_df = engineer_features(test_df)
    
    # Prepare test features
    X_test = test_df[feature_cols]
    
    # Predict fraud for test data
    predictions = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'is_fraud': predictions
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created successfully!")
    print("Submission Preview:")
    print(submission.head())
    print(f"\nTotal predictions: {len(submission)}")
    print(f"Fraudulent transactions predicted: {submission['is_fraud'].sum()}")

if __name__ == '__main__':
    main()