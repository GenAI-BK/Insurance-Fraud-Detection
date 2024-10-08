import PyPDF2
from openai import OpenAI
import PyPDF2
import streamlit as st
import pandas as pd
import sqlite3
import base64

llm=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

conn = sqlite3.connect('claim.db')
cursor = conn.cursor()

policyholder_docs = {
    "John Doe": "Policyholder_Document_John.pdf",
    "Emily Smith": "Policyholder_Document_Emily.pdf",
    "Micheal Townley": "Policyholder_Document_Micheal.pdf",
    "Franklin Johnson": "Policyholder_Document_Franklin.pdf",
    "Francis DeMaria": "/Policyholder_Document_Francis.pdf",
    "Trevor Andre": "Policyholder_Document_Trevor.pdf"
    # Add more mappings as needed
}

def convert_pdf_to_text(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    return text

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
    return pdf_display

cursor.execute("SELECT DISTINCT Name FROM claim_data")
names = [row[0] for row in cursor.fetchall()]

st.title("Insurance Claim and Fraud Detection")

selected_name = st.selectbox("Select Policyholder Name", names)

if selected_name:
    cursor.execute("SELECT Policy_No FROM claim_data WHERE Name = ?", (selected_name,))
    policy_number = cursor.fetchone()
    cursor.execute('SELECT * FROM claim_data WHERE Policy_No = ?', (policy_number))
    final=cursor.fetchall()
    
    if policy_number:
        st.write(f"Policy Number: {policy_number[0]}")
    else:
        st.write("No policy number found for the selected name.")
    
    if selected_name in policyholder_docs:
        policy_document_path = policyholder_docs[selected_name]
        st.markdown(display_pdf(policy_document_path), unsafe_allow_html=True)
        policy_terms_text=convert_pdf_to_text(policy_document_path) 
    else:
        print("No document found for the selected policyholder.")

# Create two file uploaders
claim_form = st.file_uploader("Upload Claim Form", type="PDF")
medical_bills=st.file_uploader("Upload Bills in a combined PDF", type="PDF")


def gen_res(policy_terms_text, claim_form_text, final, bills):
    response=llm.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role":"user", "content":f'''You are an employee of claims department of a health insurance company.\
        The data provided to you is the data of a policyholder along with claim form and policyholder\
        terms and conditions. Firstly, check if all the necessary data is filled in the claim form,\
        which includes, Policy number, Policyholder name, Insured Members details, Date, Patient's name,\
        services cost, etc. The claim form information must match with Policy terms information. \
        The amount mentioned in the claim form must be compared with the Bills and Policyholder \
        terms and conditions, the claim amount must not exceed the amount in Policyholder terms and conditions\
        If the claim amount is within limit, then check the bills and compare if every amount is written correctly. \
        If any of the necessary information is missing, stop the process and say "The claim form\
        is incomplete. Please upload the filled form to proceed". If the data is not missing,\
        Take into consideration all the historical data provided for the policy holder and compare claim form with\
        policyholder terms and suggest if the claim should be accepted or rejected. Also provide\
        percentage of that claim being fraud if the claim is potential fraud. \
        The historical data is provided after extracting from a SQL database with columns,\
        'Name', 'Policy_No', 'Amount', 'Date', 'Claim_Status, 'Claim_Type', 'Comments'. \
        Keep the answer crisp and short. \
        Justify your fraud percentage in 2-3 bullet points.
        
        
        Policy Terms: {policy_terms_text} 
        Claim Form: {claim_form_text}
        Data:{final}
        Bills: {bills}
        '''}],
    )
    return response.choices[0].message.content

if st.button('Submit'):
    if claim_form and medical_bills is not None and policy_terms_text:
        bills=convert_pdf_to_text(medical_bills)
        
        claim_form_text = convert_pdf_to_text(claim_form)
        
        ans=gen_res(policy_terms_text, claim_form_text, final, bills)
        print(final)
        print(bills)
        print(claim_form_text)
        print(policy_terms_text)
        st.write(ans)
    else:
        st.write("Please upload all files to proceed.")
