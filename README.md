# HawkEye: Fraud Detection System

HawkEye is an advanced fraud detection model designed to identify fraudulent transactions with high precision using **XGBoost** and **GPU acceleration**. This repository contains the implementation of HawkEye along with benchmark results.

## 🚀 Features
- **Optimized Preprocessing:** Efficient feature engineering with categorical encoding, numerical scaling, and high-risk country detection.
- **GPU-Accelerated Training:** Uses **XGBoost** with CUDA support for faster model training.
- **Fraud Detection Analysis:** Provides performance benchmarks, confusion matrix, and business impact metrics.
- **Threshold Optimization:** Finds the best classification threshold using **F1-score**.

## 📊 Benchmark Results

| Metric          | Value  |
|----------------|--------|
| **ROC AUC**    | 0.7729 |
| **Accuracy**   | 93.2%  |
| **F1 Score**   | 0.3361 |
| **Precision**  | 0.3392 |
| **Recall**     | 0.3331 |

🔹 **Total Fraudulent Transactions:** 1222  
🔹 **Detected Fraudulent Transactions:** 407  
🔹 **Detection Rate:** 33.3%  
🔹 **False Positive Rate:** 3.4%

## 📂 Repository Structure
```
HawkEye-Fraud-Detection/
│-- datasets/                     # Sample datasets (not included in repo)
│-- models/                        # Saved trained models
│-- src/                           # Source code for training and evaluation
│-- README.md                      # Project documentation
│-- requirements.txt                # Dependencies
```

## 🛠 Installation

Clone the repository and install dependencies:
```bash
$ git clone https://github.com/yourusername/HawkEye-Fraud-Detection.git
$ cd HawkEye-Fraud-Detection
$ pip install -r requirements.txt
```

## 🚀 Training the Model
To train the model using a dataset:
```bash
$ python src/train_model.py
```
This will process the dataset, train the model, and save it as `deepfraud_model.json`.

## 📈 Evaluating the Model
After training, evaluate the model's performance:
```bash
$ python src/evaluate_model.py
```
This script provides classification reports, confusion matrix, and an **ROC curve** visualization.

## 💾 Saving & Loading the Model
The trained model is saved as `deepfraud_model.json`. To load the model for predictions:
```python
import xgboost as xgb
model = xgb.Booster()
model.load_model("models/deepfraud_model.json")
```

## 📌 Contributing
Feel free to fork this repo and submit pull requests with improvements!

## 📜 License
This project is licensed under the MIT License.

## 📩 Contact
For any queries, reach out via GitHub Issues or email: **nihad.shukurlu11@gmail.com**

