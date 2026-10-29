Credit Card Fraud Detection System
A machine learning project that predicts whether a credit card
transaction may be fraudulent. The project includes a Jupyter Notebook
for experimentation, a Streamlit application for interactive
predictions, trained model files, and CSV files containing evaluation
results.
Project Overview
Credit card fraud detection is a binary classification problem. The goal
is to identify potentially fraudulent transactions while limiting
incorrect fraud alerts.

This repository contains a Streamlit interface that lets users select a
trained model and upload transaction data for prediction.
Features
- Interactive Streamlit application (app.py)
- Random Forest and XGBoost model options
- Upload a CSV file to generate predictions
- Saved model and scaler files
- Jupyter Notebook containing the project workflow
- CSV outputs for model metrics, confusion matrix, classification
  report, ROC curve, feature importance, and prediction results
Technology Stack
- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- Streamlit
- joblib
- Jupyter Notebook
- Power BI (a separate report was created locally; the .pbix file is
  not included in this repository)
Repository Files
  File                                  Purpose
  app.py                              Streamlit application

  Credit Card Fraud Detection.ipynb   Notebook for the machine learning
                                        workflow

  requirements.txt                    Python dependencies

  random_forest_model.pkl             Saved Random Forest model

  xgboost_fraud_model.pkl             Saved XGBoost model

  scaler.pkl                          Saved feature scaler used by the
                                        project

  Credit_Card_Model.pkl /             Additional saved model artifacts
  Credit_Card_Model_1.pkl             

  model_metrics.csv                   Model evaluation metrics

  classification_report.csv           Precision, recall, F1-score and
                                        support report

  confusion_matrix.csv                Confusion matrix results

  comparision.csv                     Model comparison results

  feature_importance.csv              Feature-importance output

  roc_curve.csv                       ROC-curve data

  prediction_results.csv              Actual labels, predicted labels,
                                        and/or fraud probabilities,
                                    depending on the generated output
Dataset note: The original creditcard.csv dataset is not
included in this repository because of its size. If you want to rerun
the notebook, obtain the dataset from a legitimate source and place it
in the location expected by the notebook. The Streamlit app can accept
a CSV file through its upload interface.

Getting Started
1. Clone the repository
git clone https://github.com/Lakshita-sharma-19/Machine-Learning-Project.git
cd Machine-Learning-Project
2. Create and activate a virtual environment (recommended)
Windows:
python -m venv .venv
.venv\Scripts\activate
macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit app
streamlit run app.py
Streamlit will print a local URL in the terminal. Open that URL in your
browser.
Using the Application
1. Start the app using the command above.
2. Choose the available model in the sidebar (Random Forest or
   XGBoost).
3. Upload a CSV file containing transaction features expected by the
   selected model.
4. Review the prediction output and any fraud probabilities provided by
   the app.
Important: The input CSV must contain the feature columns and
compatible preprocessing expected by the trained model. Check app.py
for the exact expected column names and preprocessing steps. Do not
include the target column (Class) as an input feature unless the app
explicitly handles it.
Evaluation Outputs
The repository includes saved CSV outputs for reviewing model
performance and predictions. Metrics such as accuracy, precision,
recall, F1-score, and ROC-AUC can be useful, but fraud datasets are
often highly imbalanced. For this reason, evaluate precision and recall
for the fraud class as well as overall accuracy.
Important Limitations
- This is an educational machine learning project, not a production
  payment-security system.
- Predictions are estimates and should not be treated as proof that a
  transaction is fraudulent or legitimate.
- Results depend on the training data, feature format, preprocessing,
  and decision threshold.
- The dataset and Power BI report file are not included in this
  repository.
Future Improvements
- Add a clear data dictionary and dataset source citation.
- Add automated tests for input validation and prediction logic.
- Compare models using precision, recall, F1-score, and ROC-AUC on a
  fixed test set.
- Improve handling of missing values and unexpected input columns.
- Deploy the Streamlit app to a hosting platform.
Author
Lakshita Sharma

MCA Student | Aspiring Data Scientist
License
No license has been specified yet. If you intend others to reuse or
modify this project, consider adding an appropriate open-source license.
