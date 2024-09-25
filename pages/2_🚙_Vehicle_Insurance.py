import streamlit as st
from openai import OpenAI
import PyPDF2
import base64
import faiss
import os
import numpy as np
import pickle
import requests
import json

st.set_page_config(page_title="Vehicle Insurance", page_icon="🚙")
st.title("Vehicle Insurance Claim and Fraud Detection")

# Initialize OpenAI API client
llm = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

INDEX_FILE_PATH = 'faiss_claim_vehicle.bin'
EMBEDDINGS_FILE_PATH = "claim_embeddings_vehicle.pkl"

# Create a new FAISS index for the session
def create_new_faiss_index():
    index = faiss.IndexFlatL2(1536)  # Create a new FAISS index
    doc_embeddings = []  # Store the document embeddings for the current session
    return index, doc_embeddings

# Embed the uploaded documents into the FAISS index for the session
# Now it will tag embeddings with filenames
def embed_uploaded_documents(uploaded_files, index, doc_embeddings):
    current_filenames = []  # Track current filenames
    for file_name, uploaded_file in uploaded_files.items():
        if uploaded_file:
            file_text = convert_pdf_to_text(uploaded_file)
            file_embedding = get_embedding(file_text)
            index.add(np.array([file_embedding]))
            # Store the filename, file content, and its embedding
            doc_embeddings.append((file_name, uploaded_file, file_embedding))
            current_filenames.append(file_name)
            st.write(f"{file_name} embedded successfully!")
    return index, doc_embeddings, current_filenames

# Function to get OpenAI embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return llm.embeddings.create(input=[text], model=model).data[0].embedding

# Convert PDF to text
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

def encode_image(uploaded_file):
    img_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_image

def ocr(image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
    }
    base64_image = encode_image(image)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Perform OCR and analyze which parts of the vehicle are damaged. \
                    Consider the right-hand side and left-hand side from the vehicle's point of view instead of the viewer's point of view. \
                    Give a list of damaged parts."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "auto"}},
                ],
            }
        ],
    })

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error processing image: {response.text}"

def vehicle_claim(claim_form, policy_terms, img1, img2, rc, dl, date, loss_type):
    response = llm.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": f'''
        Imagine you are a Vehicle insurance claims department employee tasked with analyzing the documents provided for a claim. 
        Based on your analysis, provide a percentage estimate of the likelihood that the claim is fraudulent, supported by specific justifications. 
        Summarize the claim details, point out any discrepancies, and check if the total claim amount is within the policy limits.

        Claim Form: {claim_form}
        Policy Terms: {policy_terms}
        Date of incident: {date}
        Vehicle Condition - Front: {img1}
        Vehicle Condition - Back: {img2}
        Type of loss: {loss_type}
        Registration Certificate: {rc}
        Driver's License: {dl}
        '''}]
    )

    summary = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate a short summary for the following text: {response.choices[0].message.content}"}]
    )
    
    return summary.choices[0].message.content

# Initialize FAISS index for session only, using distinct keys for vehicle insurance
if 'vehicle_index' not in st.session_state:
    st.session_state['vehicle_index'], st.session_state['vehicle_doc_embeddings'] = create_new_faiss_index()

# Track embedded documents for vehicle insurance
if 'vehicle_summary' not in st.session_state:
    st.session_state['vehicle_summary'] = None

if 'vehicle_chat_history' not in st.session_state:
    st.session_state['vehicle_chat_history'] = []

# Sample policyholders and their corresponding document paths for vehicle insurance
policyholder_docs = {
    "Riya Malhotra": "Vehicle_Policyholder/Riya_Malhotra.pdf",
    "Aarav Kapoor": "Vehicle_Policyholder/Aarav_Kapoor.pdf",
    "Kavya Singh": "Vehicle_Policyholder/Kavya_Singh.pdf",
    "Ishan Deshmukh": "Vehicle_Policyholder/Ishan_Deshmukh.pdf",
    "Meera Patel": "Vehicle_Policyholder/Meera_Patel.pdf",
}

# Select policyholder
selected_name = st.selectbox("Select Policyholder Name", list(policyholder_docs.keys()))

# Display the PDF corresponding to the selected policyholder
if selected_name:
    policy_document_path = policyholder_docs[selected_name]
    st.markdown(display_pdf(policy_document_path), unsafe_allow_html=True)
    policy_terms_text = convert_pdf_to_text(policy_document_path)

# File uploaders
incident_date = st.date_input("Date of Accident/Occurrence", value=None, format="DD/MM/YYYY")
loss_type = st.multiselect("Type of loss", ["Damage", "Theft", "Third Party"])
claim_form = st.file_uploader("Upload Claim Form", type="PDF")
rc = st.file_uploader("Upload Vehicle RC", type=['pdf'])
dl = st.file_uploader("Upload Driver's License", type=['pdf'])
vehicle_image_front = st.file_uploader("Upload Vehicle front image", type=['jpg', 'png'])
vehicle_image_back = st.file_uploader("Upload Vehicle back image", type=['jpg', 'png'])

# Store current filenames for current session
if 'vehicle_current_filenames' not in st.session_state:
    st.session_state['vehicle_current_filenames'] = []

# Embed documents and submit the claim
if st.button('Submit'):
    if claim_form and incident_date and loss_type and vehicle_image_front and vehicle_image_back:
        uploaded_files = {
            "Claim Form": claim_form,
            "Registration Certificate": rc,
            "Driver's License": dl
        }
        
        # Embed the new uploaded documents
        index, doc_embeddings, current_filenames = embed_uploaded_documents(
            uploaded_files, 
            st.session_state['vehicle_index'], 
            st.session_state['vehicle_doc_embeddings']
        )
        
        # Save the updated index and embeddings back to session state
        st.session_state['vehicle_index'] = index
        st.session_state['vehicle_doc_embeddings'] = doc_embeddings
        st.session_state['vehicle_current_filenames'] = current_filenames
        
        st.write("New documents have been successfully embedded.")

        vehicle_front_text = ocr(vehicle_image_front)
        vehicle_back_text = ocr(vehicle_image_back)

        try:
            front_image_result = json.loads(vehicle_front_text)
            st.json(front_image_result)
        except json.JSONDecodeError:
            front_image_result = vehicle_front_text

        try:
            back_image_result = json.loads(vehicle_back_text)
            st.json(back_image_result)
        except json.JSONDecodeError:
            back_image_result = vehicle_back_text

        claim_form_text = convert_pdf_to_text(claim_form)
        rc_text = convert_pdf_to_text(rc)
        dl_text = convert_pdf_to_text(dl)
        st.session_state['vehicle_summary'] = vehicle_claim(claim_form_text, policy_terms_text, front_image_result, back_image_result, rc_text, dl_text, incident_date, loss_type)
        st.write("Summary of response:", st.session_state['vehicle_summary'])

# Restrict retrieval to documents uploaded in the current session
def retrieve_and_answer(query, index, doc_embeddings, current_filenames):
    if len(doc_embeddings) == 0:
        return "No documents have been embedded yet. Please upload documents to ask queries."

    # Get the embedding for the user's query
    query_embedding = get_embedding(query)

    # Search the FAISS index using the query embedding
    D, I = index.search(np.array([query_embedding]), k=3)  # Adjust k for the number of top results you want

    # Filter document embeddings based on the current session's filenames
    filtered_results = []
    for i in I[0]:
        if i >= 0 and i < len(doc_embeddings):
            filename, file, embedding = doc_embeddings[i]
            if filename in current_filenames:
                filtered_results.append((filename, file, embedding))

    # Check if we found relevant documents
    if len(filtered_results) == 0:
        return "No relevant documents found for your query."

    # Convert the retrieved documents to text and form a context for answering
    context = " ".join([convert_pdf_to_text(file) for _, file, _ in filtered_results])

    # Use OpenAI's LLM to generate a response using the retrieved context
    response = llm.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": f"Using the following context: {context}\nAnswer: {query}"}]
    )

    return response.choices[0].message.content


# Handle further questions from the user
query = st.chat_input("Ask further questions about the claim:")
if query:
    index = st.session_state.get('vehicle_index', None)
    doc_embeddings = st.session_state.get('vehicle_doc_embeddings', [])
    current_filenames = st.session_state.get('vehicle_current_filenames', [])

    if index and doc_embeddings:
        answer = retrieve_and_answer(query, index, doc_embeddings, current_filenames)
        if st.session_state['vehicle_summary']:
            st.write("Summary", st.session_state['vehicle_summary'])
        st.session_state['vehicle_chat_history'].append((query, answer))

    if st.session_state['vehicle_chat_history']:
        for i,(q,a) in enumerate(st.session_state['vehicle_chat_history']):
            st.chat_message("User").write(q)
            st.chat_message("Assistant").write(a)
    else:
        st.write("No documents have been embedded yet. Please upload documents to ask queries.")
