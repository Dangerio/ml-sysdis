import re
import gradio as gr
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

model = CatBoostClassifier()
model.load_model('catboost_fraud_model.cbm')

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

def predict_fraud(title, description, company_profile):
    input_data = pd.DataFrame({
        'title_processed': [title],
        'description_processed': [description],
        'company_profile_processed': [company_profile]
    })
    for feat in input_data.columns:
        input_data[feat] = input_data[feat].apply(preprocess_text)
    
    
    prediction_proba = model.predict_proba(input_data)[0]
    fraud_probability = round(float(prediction_proba[1]), 2)
    
    return (
        fraud_probability,
        "Fraud" if fraud_probability > 0.5 else "Not Fraud"
    )

demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Textbox(label="Job Title"),
        gr.Textbox(label="Description", lines=5),
        gr.Textbox(label="Company Profile", lines=3)
    ],
    outputs=[
        gr.Number(label="Fraud Probability"),
        gr.Textbox(label="Prediction")
    ],
    title="Job Posting Fraud Detector",
    description="Detect potentially fraudulent job postings using CatBoost",
    examples=[
        ["Oil Rig Maintenance Technician", "High salary position ($15k/month) with no experience required. Must pay $500 security deposit for equipment. Visa processing fees apply. Start immediately!", "Offshore drilling consortium"],
        ["Senior Software Engineer", "Looking for experienced Python developer with 5+ years experience.", "Established tech company with 1000+ employees"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_port=8000, share=True)