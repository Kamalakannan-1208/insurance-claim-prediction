# requirements python, numpy, pandas, matplotlib, seaborn, joblib, streamlit, openpyxl
# to run this "streamlit run deploy.py" or streamlit run file_name.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Medical Claim Predictor")

upload_file = st.file_uploader("Upload file(CSV or Excel)", type=["csv", "xlsx"])

if upload_file is not None:
    if upload_file.name.endswith('.csv'):
        new_data = pd.read_csv(upload_file)
    else:
        new_data = pd.read_excel(upload_file)

    data = new_data.copy()
    data['Payment Amount'] = data['Payment Amount'].replace('[\$,]', '', regex=True).astype(float)
    data['Balance'] = data['Balance'].replace('[\$,]', '', regex=True).astype(float)
    data = data.drop(columns=["Denial Reason", "#", "Physician Name"], errors='ignore')
    data = pd.get_dummies(data)
    
    model_columns = joblib.load("model_columns.pkl")
    data = data.reindex(columns=model_columns, fill_value=0)

    predict = model.predict(data)
    label_map = {1: 'Denied', 0: 'Not Denied'}
    new_data['Predicted Denial'] = [label_map[i] for i in predict]

    st.write("### Prediction Results:")
    st.dataframe(new_data)

