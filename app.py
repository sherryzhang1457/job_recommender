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

#--------------------------------------------LLM (Gemini pro) API-----------------------------------------------------------#
# load gemini pro LLM model API from environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    generation_config = {
        "temperature": 0.1
    }
    model=genai.GenerativeModel(model_name = 'gemini-pro',
                                generation_config = generation_config
                            )
    response=model.generate_content([input,pdf_content,prompt])
    return response.text

# Generate prompts to generate resume revision and cover letter template

input_prompt_resume1 = """
You are an skilled Applicant Tracking System scanner with a deep understanding of Applicant Tracking System functionality, 
your task is to evaluate the resume against the provided job description. 
Find out the requirements that make this resume disqualified for this job in a list. 
Please limit the list up to five most important bullet points and no more than 30 words for each bullet points.
"""

input_prompt_resume2 = """
You are submitting a resume to a job with the provided job description. 
Find out the requirements in the job description you should add to make you qualify for this job.
Please limit the list up to five most important bullet points.
"""

input_prompt_cover_letter = """
You are the applicant who applied for this job and want to compose a strong but concise cover letter to convince the employer
you have the skills and the expereince for this job.
The first paragraph of the  cover letter must briefly discuss the your background, including both experience and projects. 
The second paragraph discuss how the applicant fit this role based on your skillsets matches the job requirements. 
Do not inlude the skillset not in the applicant's resume.
The third paragraph discuss the your interest in this role and thanks for the consideration.
Please limit the word count of cover letter no more than 300 words.
"""
#---------------------------------------------------Vector Database-------------------------------------------------------#

# Get vector database collection from local storage
chroma_client = chromadb.PersistentClient(path='db/')
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(name="job_postings")

# Find the most relevant job description and return the job posting information 
def get_relevant_ids(query, db, count=3, citizen_required = False and True, year_max = 30):
    passage = db.query(query_texts=[query],
                     n_results=count, 
                     include = ["distances", "documents", "metadatas"],
                     where={    
                       "$and": [
                      {
                          "citizen": {
                              "$eq": citizen_required
                          }
                      },
                      {
                          "minimum": {
                              "$lte": year_max
                          }
                      }
                     ] }
                     )
    ids = passage['ids'][0]
    cos = passage['distances'][0]
    doc = passage['documents'][0]
    metadata = passage['metadatas'][0]
    return ids, cos, doc, metadata

# Upload resume
resume = ''
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#---------------------------------------------------Website---------------------------------------------------------#
# Page setup

st.title("Data Science Job Matching and Resume Enhancement")
st.markdown("Powered by Gemini Pro and Chroma vector database to help you find the most relevant \
         job openings and provide specific resume revision suggestion and cover letter template.")
st.markdown("Please be patient while waiting for the LLM-generated suggestions.") 
st.divider()

# Sidebar for user interaction
submit = None
with st.sidebar:
    uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf, the app won't save your resume")
  
    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")
        resume = input_pdf_text(uploaded_file)

    result_count = st.number_input('Results count', 1, 100, 30)
    st.write('')

    citizenship_included = st.checkbox('Include US citizen only job')
    if citizenship_included:
        citizen_required = False
    else:
        citizen_required = False and True

    year_max = st.slider('Years of experience range', 0, 30, 0)

    if resume != '':
        submit = st.button("Generate LLM-powered results")

# Show results
if submit:
# Perform embedding search with vector database
    results, score, doc, meta = get_relevant_ids(resume, collection, result_count, citizen_required, year_max)
    
    with st.container():
        for i in range(len(results)):
            with st.expander(meta[i]['info']):
                st.markdown(f'Similarity score: %.2f' %(1 - score[i]))
                st.markdown('**Job Description**')
                st.write(doc[i])
                st.link_button("Apply it!", meta[i]['link'], type="primary")

                response=get_gemini_response(input_prompt_resume1,resume,doc[i])
                st.subheader("Disqualifications")
                st.write(response)        

                response=get_gemini_response(input_prompt_resume2,resume,doc[i])
                st.subheader("Skills you may want to add")
                st.write(response)

                response=get_gemini_response(input_prompt_cover_letter,resume,doc[i])
                st.subheader("Coverletter")
                st.write(response)
                time.sleep(2)

                if i % 10 == 0:
                    time.sleep(5)
