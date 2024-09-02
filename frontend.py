# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
import os
import PyPDF2
from PIL import Image
import streamlit as st
import pandas as pd
import sqlite3
import io
import base64
import pdfplumber

api_key = st.secrets["GOOGLE_API_KEY"]

conn = sqlite3.connect('claim.db')
cursor = conn.cursor()

genai.configure(api_key=api_key)
llm = genai.GenerativeModel("gemini-1.5-pro")

policyholder_docs = {
    "John Doe": "Policyholder_Document_John.pdf",
    "Emily Smith": "/Policyholder_Document_Emily.pdf",
    "Micheal Townley": "/Policyholder_Document_Micheal.pdf",
    "Franklin Johnson": "/Policyholder_Document_Franklin.pdf",
    "Francis DeMaria": "/Policyholder_Document_Francis.pdf",
    "Trevor Andre": "/Policyholder_Document_Trevor.pdf"
    # Add more mappings as needed
}

def excel_to_db(csv_file, db_file, table_name):
    # Read CSV file into a pandas DataFrame
    df = pd.read_excel(csv_file)
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    # Write DataFrame to SQLite database
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    # Close the connection
    conn.close()

def convert_pdf_to_text(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    return text

def excel_to_text(xls):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_excel('John_Doe_Premium_Payment.xlsx')

    # Select all columns and convert it to a text Series
    text_series_list = [df[col].astype(str) for col in df.columns]

    # Join the text Series into a single string
    text_strings = [' '.join(text_series) for text_series in text_series_list]
    for text_string in text_strings:
        xls_text=text_string
    return xls_text

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
    return pdf_display

def extract_text_and_images(pdf_path):
    elements = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text and preserve layout
            text = page.extract_text()
            if text:
                elements.append(('text', text))

            # Extract images
            for image in page.images:
                # Image metadata
                x0 = image['x0']
                top = image['top']
                x1 = image['x1']
                bottom = image['bottom']
                
                # Extract image as bytes
                img_data = page.to_image().original
                img_data = img_data.crop((x0, top, x1, bottom))
                
                # Convert cropped image to a PIL image
                img_bytes = io.BytesIO()
                img_data.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Open image with PIL
                image_pil = Image.open(img_bytes)
                elements.append(('image', image_pil))

    return elements

if 'pdf1' not in st.session_state:
    st.session_state.pdf1 = None
if 'pdf2' not in st.session_state:
    st.session_state.pdf2 = None

cursor.execute("SELECT DISTINCT Name FROM claim_data")
names = [row[0] for row in cursor.fetchall()]

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
    # if selected_name in policyholder_docs:
    #     policy_document_path = policyholder_docs[selected_name]
        policy_terms_text=convert_pdf_to_text(policy_document_path)
    
        # with open(policy_document_path, "rb") as policy_file:
        #     pdf_bytes = policy_file.read()
        #     st.download_button(
        #         label=f"Download {selected_name}'s Document",
        #         data=pdf_bytes,
        #         file_name=f"{selected_name}_Document.pdf",
        #         mime="application/pdf",
        #     )
        
    else:
        print("No document found for the selected policyholder.")

# Create two file uploaders
claim_form = st.file_uploader("Upload Claim Form", type="PDF")
# uploaded_pdf2 = st.file_uploader("Upload Policy holder Document", type="pdf")
medical_bills=st.file_uploader("Upload Bills in a combined PDF", type="PDF")


def gen_res(policy_terms_text, claim_form_text, final, bills):
    response=llm.generate_content([f'''You are an employee of claims department of a health insurance company.\
        The data provided to you is the data of a policyholder along with claim form and policyholder\
        terms and conditions. Firstly, check if all the necessary data is filled in the claim form,\
        which includes, Policy number, Policyholder name, Insured Members details, Date, Patient's name,\
        services cost, etc. The claim form information must match with Policy terms information \
        If any of the necessary information is missing, stop the process and say "The claim form\
        is incomplete. Please upload the filled form to proceed". If the data is not missing,\
        Take into consideration all the historical data provided for the policy holder and compare claim form with\
        policyholder terms and suggest if the claim should be accepted or rejected. Also provide\
        percentage of that claim being fraud if the claim is potential fraud
        If the previous data is not available for any\
        particular claimant, try to compare the cases of fraudery in same 'Claim_Type' column.\
        Keep the answer crisp and short. \
        Justify your fraud percentage in 2-3 bullet points
        
        
        Policy Terms: {policy_terms_text} 
        Claim Form: {claim_form_text}
        Data:{final}
        Bills: {bills}
        ''']
        )
    return response.text


if st.button('Submit'):
    bills=str(extract_text_and_images(medical_bills))
    if claim_form and medical_bills is not None and policy_terms_text:
        claim_form_text = convert_pdf_to_text(claim_form)
        ans=gen_res(policy_terms_text, claim_form_text, final, bills)
        # bills= convert_pdf_to_text(medical_bills)

        # claim_form_text = convert_pdf_to_text(st.session_state.pdf1)
        # policy_terms_text = convert_pdf_to_text(st.session_state.pdf2)

    else:
        st.write("Please upload all files to proceed.")
