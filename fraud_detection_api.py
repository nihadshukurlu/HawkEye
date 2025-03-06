from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import shap
import numpy as np

# Modeli yükle
search_model = joblib.load("fraud_detection_model.pkl")
model_pipeline = search_model.best_estimator_  # Pipeline içindeki en iyi modeli al

# Pipeline içinden scaler ve sınıflandırıcıyı al
scaler = model_pipeline.named_steps['scaler']
classifier_model = model_pipeline.named_steps['clf']  # AdaBoostClassifier

# FastAPI uygulaması oluştur
app = FastAPI()

# Modelin eğitildiği tam feature set'i al
trained_features = search_model.feature_names_in_.tolist()

# 📌 Load past fraud transactions from CSV with correct column names
fraud_data_path = "Fraudulent_E-Commerce_Transaction_Data_2.csv"  # CSV file name
fraud_df = pd.read_csv(fraud_data_path)

# Rename columns to match API expectations
fraud_df = fraud_df.rename(columns={
    "Transaction Amount": "txn_amt",
    "Payment Method": "payment_method",
    "Product Category": "prod_cat",
    "Quantity": "qty",
    "Customer Age": "cust_age",
    "Device Used": "device_used",
    "Account Age Days": "acct_age_days",
    "Transaction Hour": "txn_hour",
    "Is Fraudulent": "fraudulent"
})

fraud_df = fraud_df[fraud_df["fraudulent"] == True]  # Filter only fraudulent transactions

# 📌 Compute fraud rate statistics for each category
fraud_rates = {
    "payment_method": fraud_df["payment_method"].value_counts(normalize=True).to_dict(),
    "prod_cat": fraud_df["prod_cat"].value_counts(normalize=True).to_dict(),
    "device_used": fraud_df["device_used"].value_counts(normalize=True).to_dict()
}


# SHAP KernelExplainer kullan (predict_proba ile çalıştırıyoruz)
def predict_proba_df(X):
    X_df = pd.DataFrame(X, columns=trained_features)
    return classifier_model.predict_proba(scaler.transform(X_df))


try:
    background_data = np.random.rand(5, len(trained_features))  # Rastgele 5 örnek al
    explainer = shap.KernelExplainer(predict_proba_df, background_data)
except Exception as e:
    explainer = None
    print(f"SHAP explainer oluşturulamadı: {e}")


# JSON şeması tanımla
class TransactionInput(BaseModel):
    txn_amt: float
    payment_method: str
    prod_cat: str
    qty: int
    cust_age: int
    device_used: str
    acct_age_days: int
    txn_hour: int


@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    # Gelen veriyi DataFrame'e çevir
    input_data = pd.DataFrame([transaction.dict()])

    # One-Hot Encoding uygulayarak eksik sütunları tamamla
    input_data = pd.get_dummies(input_data, columns=["device_used", "payment_method", "prod_cat"])

    # Eksik veya fazla sütunları yönet
    for col in trained_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Eksik sütunları 0 ile doldur

    # Fazla sütunları kaldır
    input_data = input_data[trained_features]

    # Scaler ile veriyi dönüştür
    input_data_scaled = scaler.transform(input_data)

    # Model tahmini yap
    prediction = classifier_model.predict(input_data_scaled)
    is_fraudulent = bool(prediction[0])

    # SHAP ile açıklama üret
    explanation = {}
    textual_explanation = []

    if explainer:
        shap_values = explainer.shap_values(input_data_scaled)
        shap_values = np.array(shap_values[0]).flatten().tolist()  # SHAP çıktısını düz listeye çeviriyoruz
        for feature, shap_value in zip(trained_features, shap_values):
            explanation[feature] = float(shap_value)
            if shap_value > 0:
                textual_explanation.append(f"{feature.replace('_', ' ').title()} nedeniyle risk arttı.")
            elif shap_value < 0:
                textual_explanation.append(f"{feature.replace('_', ' ').title()} nedeniyle risk azaldı.")

    # 📌 Find similar past fraud transactions only if the model predicts fraud
    similar_frauds = []
    if is_fraudulent:
        similar_frauds = fraud_df[
            (fraud_df["payment_method"] == transaction.payment_method) &
            (fraud_df["device_used"] == transaction.device_used) &
            (fraud_df["prod_cat"] == transaction.prod_cat) &
            (abs(fraud_df["txn_amt"] - transaction.txn_amt) / transaction.txn_amt < 0.2)  # %20 similarity threshold
            ]
        similar_frauds = similar_frauds.to_dict(orient="records")  # Convert to JSON format

    # 📌 Add fraud probability for selected payment method, product category, and device
    fraud_risk_factors = {
        "payment_method_risk": fraud_rates["payment_method"].get(transaction.payment_method, 0.0),
        "prod_cat_risk": fraud_rates["prod_cat"].get(transaction.prod_cat, 0.0),
        "device_used_risk": fraud_rates["device_used"].get(transaction.device_used, 0.0)
    }

    return {
        "fraudulent": is_fraudulent,
        "explanation": explanation if explanation else "SHAP explainer oluşturulamadı.",
        "textual_explanation": textual_explanation,
        "similar_fraud_transactions": similar_frauds[:5],  # Show top 5 similar fraud cases only if fraudulent
        "fraud_risk_factors": fraud_risk_factors  # Add fraud probability info
    }