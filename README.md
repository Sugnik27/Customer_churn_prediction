# Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company using various classification algorithms.

Live App Link: https://customerchurnprediction-gnopmjyzysnuhb7vuoh7v5.streamlit.app/#enter-customer-details

## 📋 Project Overview

This project analyzes customer data to predict whether a customer is likely to churn (leave the service). The model helps businesses identify at-risk customers and take proactive measures to retain them.

## 🎯 Features

- **Data Preprocessing**: Automated data cleaning, normalization, and feature engineering
- **Multiple ML Models**: Comparison of 5 different algorithms
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVC)
  - XGBoost
- **Hyperparameter Tuning**: Grid Search CV for optimal model selection
- **Best Model Selection**: Automatic selection based on ROC-AUC score
- **Feature Engineering**: Intelligent handling of categorical and numerical features
- **Model Persistence**: Saves the best performing model for deployment

## 📊 Dataset

The project uses the Telco Customer Churn dataset containing:
- **Customer Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Service Information**: Phone Service, Internet Service, Online Security, etc.
- **Account Information**: Tenure, Contract type, Payment Method, Monthly Charges
- **Target Variable**: Churn (Yes/No)

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Joblib**: Model serialization

## 📁 Project Structure

```
Customer_churn_prediction/
│
├── data/
│   ├── Telco_Customer_Churn.csv    # Raw dataset
│   └── cleaned_data.csv             # Preprocessed dataset
│
├── models/
│   ├── best_model.joblib            # Best performing model
│   └── feature_columns.json         # Feature column names
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Data exploration
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration parameters
│   ├── preprocessing.py             # Data preprocessing functions
│   ├── training.py                  # Model training pipeline
│   └── app.py                       # Streamlit application
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sugnik27/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv virtual_env
```

### 3. Activate the virtual environment

**Windows:**
```bash
virtual_env\Scripts\activate
```

**Linux/Mac:**
```bash
source virtual_env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

To train the model and perform hyperparameter tuning:

```bash
python -m src.training
```

This will:
1. Load and preprocess the data
2. Split data into training and test sets
3. Train multiple models with grid search
4. Select and save the best performing model
5. Display performance metrics

### Running the Streamlit App

To launch the interactive web application:

```bash
streamlit run src/app.py
```

## 📈 Model Performance

The project evaluates models based on ROC-AUC score. Current best model performance:

| Model | ROC-AUC Score |
|-------|---------------|
| XGBoost | 0.8461 |
| SVC | 0.8266 |
| Gradient Boosting | 0.8459 |
| Logistic Regression | 0.8418 |
| Random Forest | 0.8408 |

**Best Model**: XGBoost with ROC-AUC of 0.8461

## 🔧 Configuration

Key parameters can be adjusted in `src/config.py`:

```python
TEST_SIZE = 0.2          # Train-test split ratio
RANDOM_STATE = 42        # Random seed for reproducibility
CV_FOLD = 5              # Cross-validation folds
SCORING = "roc_auc"      # Model evaluation metric
N_JOBS = -1              # Parallel processing
TARGET_COLUMN = "Churn"  # Target variable name
```

## 📊 Model Pipeline

1. **Data Loading**: Load raw customer data
2. **Data Cleaning**: Handle missing values and normalize column names
3. **Feature Engineering**: 
   - Categorical encoding using OneHotEncoder
   - Numerical scaling using StandardScaler
4. **Train-Test Split**: Stratified split to maintain class distribution
5. **Model Training**: Grid search with cross-validation
6. **Model Evaluation**: ROC-AUC scoring
7. **Model Selection**: Save best performing model

## 🎓 Key Insights

- **Best Algorithm**: XGBoost performs best for this classification task
- **Important Features**: Contract type, tenure, and monthly charges are strong predictors
- **Class Balance**: The dataset shows imbalanced classes, handled through stratified splitting
- **Preprocessing Impact**: Proper encoding and scaling significantly improve model performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name**
- GitHub: [@Sugnik27](https://github.com/Sugnik27)
- Email: sugnik.official@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspiration from various Kaggle kernels and ML tutorials
- Special thanks to the scikit-learn and XGBoost communities

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

---

