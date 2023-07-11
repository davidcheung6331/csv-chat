from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
import os
import tempfile
import streamlit as st

from PIL import Image

st.set_page_config(
    page_title="Demo Page",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# image = Image.open("stock.png")
# st.image(image, caption='created by MJ')

system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key



log = ""
st.subheader("Step 1. 📤 Upload a csv file ")
uploaded_file = st.file_uploader("Select file", type=['csv'])
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    log = log + "\nTemporary Directory : " + temp_dir.name

    upload_filename = uploaded_file.name
    temp_file_path = os.path.join(temp_dir.name, upload_filename)
    log = log + "\nFull File Path : " + temp_file_path

    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())
    log = log + "\nCreate the temp file"

    
    log = log + f"\nCSVLoader load CSV : {temp_file_path}"

    loader = CSVLoader(file_path=upload_filename)   
    # loader = CSVLoader(file_path=temp_file_path)  

    index_creator = VectorstoreIndexCreator()
    log = log + "\nCreate Vectorstore Index"
    # docsearch = index_creator.from_loaders([loader])
    # docsearch = index_creator.from_loaders([loader], ids=[])
    docsearch = index_creator.from_loaders([loader] if loader else [])


    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    log = log + "\nCreate Langchain RetrievalQA "
    
    with st.expander("For details, click here"):
        st.code(log)

    query = "How many subjects?"
    query = st.text_input("Enter your question for this CSV", query)
    if st.button("Submit",  type="primary"):
        response = chain({"question": query})
        st.info(response['result'])




