__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv
import json
import os
from os import path, listdir
import numpy as np
import pandas as pd
import PyPDF2 as pdf
import streamlit as st
import google.generativeai as genai
import chromadb
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from chromadb.utils import embedding_functions


# use gemini pro LLM model API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro')
    generation_config = genai.GenerationConfig(
    temperature=0.0
    )
    response=model.generate_content([input,pdf_content,prompt],generation_config=generation_config)
    return response.text

input_prompt_resume1 = """
You are an skilled Applicant Tracking System scanner with a deep understanding of Applicant Tracking System functionality, 
your task is to evaluate the resume against the provided job description. 
Find out the requirements that make this resume disqualified for this job in a list.
"""

input_prompt_resume2 = """
You are submitting a resume to a job with the provided job description. 
Find out the requirements in the job description you should add to make you qualify for this job.
"""

input_prompt_cover_letter = """
You are the applicant who applied for this job and want to compose a strong but concise coverletter to convince the employer you have the skills and the expereince for this job.
The first paragraph of the  cover letter must briefly discuss the your backgroud. 
The second paragraph discuss how the applicant fit this role based on your skillsets matches the job requirements.
The third paragraph discuss the your interest in this role and thanks for the consideration .
"""

st.set_page_config(page_title='Job Recommender System')
data_dir = 'data/'

# Initiating a persistent Chroma client
# Data will be persisted to a local machine
# configuration = {
#     "client": "PersistentClient",
#     "path": "/tmp/.chroma"
# }

# collection_name = "documents_collection"

# conn = st.connection("chromadb",
#                      type=ChromaDBConnection,
#                      **configuration)
# documents_collection_df = conn.get_collection_data(collection_name)
# st.dataframe(documents_collection_df)
# # create a Chroma collection
# collection_name = "documents_collection"
# embedding_function_name = "DefaultEmbedding"
# conn.create_collection(collection_name=collection_name,
#                        embedding_function_name=embedding_function_name)

chroma_client = chromadb.Client()
default_ef = embedding_functions.DefaultEmbeddingFunction()
def create_chroma_db(df, name):
  db = chroma_client.create_collection(name=name, embedding_function=default_ef)

  for index, row in df.iterrows():
    db.add(
      documents=row['job_summary'],
      metadatas=[{"title": row['job_title'], 
                  "company": row['company'],
                 }],
      ids=str(index)
    )
  return db
    
# Set up the DB
job_postings = pd.read_csv('postings.csv')
job_postings = job_postings.dropna()

# collection = chroma_client.get_collection(name="jobdatabase", embedding_function=default_ef)
# if not collection:
collection = create_chroma_db(job_postings, "jobdatabase")

# Confirm that the data was inserted by looking at the database
pd.DataFrame(collection.peek(3))

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage



# def recommend_jobs(resume: str, item_count: int = 30) -> pd.DataFrame:
#     jobs_list = pd.concat(
#         [pd.Series([resume]), data_jd],
#         ignore_index=True
#     )
#     tfidf = TfidfVectorizer(stop_words='english',
#                             tokenizer=stem_tokenizer,
#                             lowercase=True,
#                             max_df=0.7,
#                             min_df=1,
#                             ngram_range=(1, 2)
#                            ).fit(data_jd)
  
#     description_matrix = tfidf.transform(jobs_list)
#     similarity_matrix = linear_kernel(description_matrix)

#     job_index = 0

#     similarity_score = list(enumerate(similarity_matrix[job_index]))
#     similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#     similarity_score = similarity_score[1:item_count + 1]

#     job_indices = [i[0] for i in similarity_score]
#     return data.iloc[job_indices]

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text
  
# data = load_data()
# data_jd = data['job_summary']
resume = ''

st.title("Data Science Job Recommender System")
with st.container():
    col1, col2, col3 = st.columns((3, 0.5, 3))

    with col1:
        uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")
      
        if uploaded_file is not None:
            st.write("PDF Uploaded Successfully")
            resume = input_pdf_text(uploaded_file)

    with col3:
        result_count = st.number_input('Results count', 1, 100, 30)
        st.write('')

if resume != '':
    # results = recommend_jobs(resume, result_count)
    # Perform embedding search
    results = get_relevant_passage(resume, collection)
    st.write(results)
    
    with st.container():
        for index, result in results.iterrows():
            with st.expander(result['job_title']):
                st.write('**Location:** ' + result['job_location'])
                st.write('**Company:** ' + result['company'])

                st.markdown('**Job Description**')
                st.write(result['job_summary'])

                st.write(f'**Link:** [{result["job_link"]}]({result["job_link"]})')

                submit = st.button("Generate LLM-powered results")
                if submit:
                    response=get_gemini_response(input_prompt_resume1,resume,result['job_summary'])
                    st.subheader("Disqualifications")
                    st.write(response)        
    
                    response=get_gemini_response(input_prompt_resume2,resume,result['job_summary'])
                    st.subheader("Skills you may want to add")
                    st.write(response)
    
                    response=get_gemini_response(input_prompt_cover_letter,resume,result['job_summary'])
                    st.subheader("Coverletter")
                    st.write(response)
                
