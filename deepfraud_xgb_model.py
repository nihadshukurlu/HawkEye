import pandas as pd
import numpy as np
import cudf
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve, accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

MODEL_TYPE = "xgboost" 
USE_DEEP = False
APPLY_SMOTE_PREPROCESSING = True   
APPLY_SMOTE_AFTER_FE = True
BEST_THRESHOLD_OVERRIDE = None

CHUNK_SIZE = 200000

HIGH_RISK_COUNTRIES = {"CountryX", "CountryY", "CountryZ"}

CATEGORICAL_FEATURES = ['Payment Method', 'Product Category',
                        'Device Used', 'Customer Location']

NUMERICAL_FEATURES = [
    'Transaction Amount', 'Quantity',
    'Customer Age', 'Account Age Days',
    'Transaction Hour', 'Transaction_Day',
    'Transaction_Month', 'Address_Mismatch',
    'AmountHourRatio', 'CustomerVelocity',
    'TransactionFrequency', 'AmountDeviation',
    'Transaction_Weekday', 'High_Risk_Country',
    'log_amount', 'is_peak_hour', 'amount_per_age'
]

def optimize_dtypes(df):
    dtype_mapping = {
        'Transaction Amount': 'float32',
        'Quantity': 'int16',
        'Customer Age': 'int16',
        'Account Age Days': 'int16',
        'Transaction Hour': 'int8',
        'Transaction_Day': 'int8',
        'Transaction_Month': 'int8',
        'Address_Mismatch': 'int8',
        'Is Fraudulent': 'bool',
        'AmountHourRatio': 'float32',
        'CustomerVelocity': 'float32',
        'TransactionFrequency': 'float32',
        'AmountDeviation': 'float32',
        'Transaction_Weekday': 'int8',
        'High_Risk_Country': 'int8',
        'log_amount': 'float32',
        'is_peak_hour': 'int8',
        'amount_per_age': 'float32'
    }
    if 'IsLargeQuantity' in df.columns:
        dtype_mapping['IsLargeQuantity'] = 'int8'
    return df.astype({k: v for k, v in dtype_mapping.items() if k in df.columns})

def preprocess(df):
    try:
        if not np.issubdtype(df['Transaction Date'].dtype, np.datetime64):
            if isinstance(df, cudf.DataFrame):
                df['Transaction Date'] = df['Transaction Date'].astype('datetime64[ms]')
            else:
                df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

        df['Transaction_Day'] = df['Transaction Date'].dt.day.astype('int8')
        df['Transaction_Month'] = df['Transaction Date'].dt.month.astype('int8')
        df['Transaction Hour'] = df['Transaction Date'].dt.hour.astype('int8')
        df['Transaction_Weekday'] = df['Transaction Date'].dt.weekday.astype('int8')

        df['Address_Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype('int8')
        df['AmountHourRatio'] = (df['Transaction Amount'] / (df['Transaction Hour'] + 1)).fillna(0).astype('float32')

        if 'Customer Location' in df.columns:
            df['High_Risk_Country'] = df['Customer Location'].isin(HIGH_RISK_COUNTRIES).astype('int8')
        else:
            df['High_Risk_Country'] = 0

        THRESHOLD_QTY = 10
        df['IsLargeQuantity'] = (df['Quantity'] > THRESHOLD_QTY).astype('int8')
        if 'IsLargeQuantity' not in NUMERICAL_FEATURES:
            NUMERICAL_FEATURES.append('IsLargeQuantity')

        if 'Customer ID' in df.columns:
            cust_stats = df.groupby('Customer ID')['Transaction Amount'].agg(['mean', 'std']).rename(
                columns={'mean': 'Cust_Trans_Mean', 'std': 'Cust_Trans_Std'}
            )
            df = df.merge(cust_stats, left_on='Customer ID', right_index=True, how='left')
            df['Cust_Trans_Std'] = df['Cust_Trans_Std'].replace(0, 1)
            df['AmountDeviation'] = ((df['Transaction Amount'] - df['Cust_Trans_Mean']) / df['Cust_Trans_Std']) \
                .fillna(0).astype('float32')

            df['CustomerVelocity'] = df.groupby('Customer ID')['Transaction Amount'].cumsum() / \
                                     (df.groupby('Customer ID').cumcount().add(1))

            df['TransactionFrequency'] = df.groupby('Customer ID')['Transaction Date'].diff() \
                                             .astype('int64') / 1e9
            df['TransactionFrequency'] = df['TransactionFrequency'].fillna(24 * 3600) / 3600
            df['TransactionFrequency'] = df['TransactionFrequency'].clip(upper=24).astype('float32')
        else:
            df['CustomerVelocity'] = 0.0
            df['AmountDeviation'] = 0.0
            df['TransactionFrequency'] = 24.0

        df['log_amount'] = np.log1p(df['Transaction Amount']).astype('float32')
        df['is_peak_hour'] = ((df['Transaction Hour'] >= 18) & (df['Transaction Hour'] <= 22)).astype('int8')
        df['amount_per_age'] = (df['Transaction Amount'] / (df['Customer Age'] + 1)).astype('float32')

    except KeyError as e:
        print(f"Missing column: {e}")
        raise

    drop_cols = ['Transaction ID', 'Customer ID', 'IP Address', 'Transaction Date', 'Shipping Address',
                 'Billing Address']
    df = df.drop(drop_cols, axis=1, errors='ignore')

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes.add(1).astype('int16')

    df = optimize_dtypes(df)

    return df

def apply_smote(X, y, stage=""):
    from imblearn.over_sampling import SMOTE
    print(f"Applying SMOTE at the {stage} stage...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_xgb_model(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'aucpr',
        'max_depth': 8,
        'learning_rate': 0.01,
        'reg_alpha': 2,
        'reg_lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
        'random_state': 42
    }
    print("Training XGBoost model on GPU...")
    model = xgb.train(
        params,
        dtrain,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        num_boost_round=500,
        early_stopping_rounds=50
    )
    print("Calculating optimal threshold using F1 Score (XGBoost)...")
    y_proba = model.predict(dval)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return model, optimal_threshold

def train_lgb_model(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'max_depth': 8,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'reg_alpha': 2,
        'reg_lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_split_gain': 0.2,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
        'device': 'gpu',
        'verbose': -1,
        'random_state': 42
    }
    print("Training LightGBM model on GPU...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    print("Calculating optimal threshold using F1 Score (LightGBM)...")
    y_proba = model.predict(X_val, num_iteration=model.best_iteration)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return model, optimal_threshold

def train_deep_model():
    print("Deep learning model integration is not implemented yet.")
    return None, 0.5

def train_model():
    DATA_PATH = "Fraudulent_E-Commerce_Transaction_Data.parquet"
    full_df = cudf.read_parquet(DATA_PATH)
    print("Data loaded into GPU memory.")

    if not np.issubdtype(full_df['Transaction Date'].dtype, np.datetime64):
        full_df['Transaction Date'] = cudf.to_datetime(full_df['Transaction Date'])
    full_df = full_df.sort_values('Transaction Date')
    split_date = full_df['Transaction Date'].quantile(0.8)

    train_df = full_df[full_df['Transaction Date'] < split_date]
    val_df = full_df[full_df['Transaction Date'] >= split_date]

    train_proc = preprocess(train_df)
    val_proc = preprocess(val_df)

    X_train = train_proc.drop('Is Fraudulent', axis=1).fillna(0).to_pandas()
    y_train = train_proc['Is Fraudulent'].to_pandas()
    X_val = val_proc.drop('Is Fraudulent', axis=1).fillna(0).to_pandas()
    y_val = val_proc['Is Fraudulent'].to_pandas()

    if APPLY_SMOTE_PREPROCESSING:
        X_train, y_train = apply_smote(X_train, y_train, stage="preprocessing")
    if APPLY_SMOTE_AFTER_FE:
        X_train, y_train = apply_smote(X_train, y_train, stage="post-feature engineering")

    if USE_DEEP:
        model, threshold = train_deep_model()
    else:
        if MODEL_TYPE == "xgboost":
            model, threshold = train_xgb_model(X_train, y_train, X_val, y_val)
        elif MODEL_TYPE == "lightgbm":
            model, threshold = train_lgb_model(X_train, y_train, X_val, y_val)
        else:
            raise ValueError("Unsupported model type")
    
    if BEST_THRESHOLD_OVERRIDE is not None:
        print(f"Overriding computed threshold with BEST_THRESHOLD={BEST_THRESHOLD_OVERRIDE}")
        threshold = BEST_THRESHOLD_OVERRIDE

    return model, threshold

def evaluate_model(model, threshold):
    TEST_CSV_PATH = '/datasets/e_commerce_transactions/Fraudulent_E-Commerce_Transaction_Data_2.csv'
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_proc = preprocess(test_df)

    X_test = test_proc.drop('Is Fraudulent', axis=1).fillna(0)
    y_test = test_proc['Is Fraudulent']

    if MODEL_TYPE == "xgboost":
        dtest = xgb.DMatrix(X_test)
        probas = model.predict(dtest)
    elif MODEL_TYPE == "lightgbm":
        probas = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        raise ValueError("Unsupported model type")

    predictions = (probas >= threshold).astype(int)

    print("\nOptimized Test Performance:")
    print(classification_report(y_test, predictions))

    total_fraud = y_test.sum()
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    detected_fraud = np.sum((y_test_np.astype(int) & predictions))
    detection_rate = detected_fraud / total_fraud if total_fraud else 0
    false_positives = predictions.sum() - detected_fraud
    false_positive_rate = false_positives / len(y_test)

    print("Business Impact Analysis:")
    print(f"- Total fraudulent transactions: {total_fraud}")
    print(f"- Detected fraudulent transactions: {detected_fraud}")
    print(f"- Detection rate: {detection_rate:.1%}")
    print(f"- False positive rate: {false_positive_rate:.1%}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, probas):.4f}")
    print(f"Optimal Threshold Used: {threshold:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, probas):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    roc_auc = roc_auc_score(y_test, probas)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("\nPerformance Metrics:")
    print(f"- ROC AUC: {roc_auc:.4f}")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

def explain_model(model, X):
    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

def save_model(model, filename="deepfraud_model.json"):
    model.save_model(filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    model, threshold = train_model()
    save_model(model, "deepfraud_model.json")
    evaluate_model(model, threshold)
