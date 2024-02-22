__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv
import json
import os, time
from os import path, listdir
import numpy as np
import pandas as pd
import PyPDF2 as pdf
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

#------------------------------------------------------------------------------------------------------------------------#
# use gemini pro LLM model API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro')
    generation_config = genai.GenerationConfig(
        temperature = 0.0,
        # max_output_tokens = 1024
    )
    # response=model.generate_content([input,pdf_content,prompt])
    response=model.generate_content([input,pdf_content,prompt],generation_config=generation_config)
    return response.text

# Generate prompts for resume revision and cover letter template
input_prompt_resume1 = """
You are an skilled Applicant Tracking System scanner with a deep understanding of Applicant Tracking System functionality, 
your task is to evaluate the resume against the provided job description. 
Find out the requirements that make this resume disqualified for this job in a list. 
Please limit the list up to five most important bullet points.
"""

input_prompt_resume2 = """
You are submitting a resume to a job with the provided job description. 
Find out the requirements in the job description you should add to make you qualify for this job.
Please limit the list up to five most important bullet points.
"""

input_prompt_cover_letter = """
You are the applicant who applied for this job and want to compose a strong but concise cover letter to convince the employer you have the skills and the expereince for this job.
The first paragraph of the  cover letter must briefly discuss the your backgroud. 
The second paragraph discuss how the applicant fit this role based on your skillsets matches the job requirements.
The third paragraph discuss the your interest in this role and thanks for the consideration.
Please limit the word count of cover letter no more than 300 words.
"""
#------------------------------------------------------------------------------------------------------------------------#

# Get vector database collection from local storage
chroma_client = chromadb.PersistentClient(path='db/')
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(name="job_postings")
    
# Read job posting dataset
job_postings = pd.read_csv('postings.csv')
job_postings = job_postings.fillna('')

# Find the most relevant job description and return the job posting information 
def get_relevant_ids(query, db, count, df):
    ids = db.query(query_texts=[query], n_results=count)['ids'][0]
    return df.iloc[ids]

# Upload resume
resume = ''
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#------------------------------------------------------------------------------------------------------------------------#
# Page setup
st.title("Data Science Job Recommender System")
submit = None
with st.sidebar:
    uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")
  
    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")
        resume = input_pdf_text(uploaded_file)

    result_count = st.number_input('Results count', 1, 100, 30)
    st.write('')

    if resume != '':
        submit = st.button("Generate LLM-powered results")
        
if submit:
# Perform embedding search with vector database
    results = get_relevant_ids(resume, collection, result_count, job_postings)
    
    with st.container():
        for index, result in results.iterrows():
            job_info = result['job_title'] + ' | ' + result['job_location'] + ' | ' + result['company'] 
            with st.expander(job_info):
                st.markdown('**Job Description**')
                st.write(result['job_summary'])
                st.link_button("Apply it!", result["job_link"], type="primary")
                # st.page_link(result["job_link"], label="**Apply it!**")

                response=get_gemini_response(input_prompt_resume1,resume,result['job_summary'])
                st.subheader("Disqualifications")
                st.write(response)        

                response=get_gemini_response(input_prompt_resume2,resume,result['job_summary'])
                st.subheader("Skills you may want to add")
                st.write(response)

                response=get_gemini_response(input_prompt_cover_letter,resume,result['job_summary'])
                st.subheader("Coverletter")
                st.write(response)
                time.sleep(5)
                
