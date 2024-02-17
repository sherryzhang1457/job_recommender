# from dotenv import load_dotenv
import json
import os
from os import path, listdir
import numpy as np
import pandas as pd
import PyPDF2 as pdf
import streamlit as st
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import spacy

from nltk.stem.snowball import EnglishStemmer
import re

# use gemini pro LLM model API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title='Job Recommender System')
data_dir = 'data/'

# Load job posting data
@st.cache(allow_output_mutation=True)
def load_data() -> pd.DataFrame:
    # csv_files = [path.join(data_dir, csv) for csv in listdir(data_dir)]
    # df = pd.concat(
    #     map(lambda csv: pd.read_csv(csv, index_col=0), csv_files),
    #     ignore_index=True
    # )
    df = pd.read_csv('postings.csv')
    df['job_summary'] = df['job_summary'].fillna('')

    return df

def stem_tokenizer(text):
    # stemmer = EnglishStemmer(ignore_stopwords=True)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    # words = [stemmer.stem(word) for word in words]
    return words


def recommend_jobs(resume: str, item_count: int = 30) -> pd.DataFrame:
    jobs_list = pd.concat(
        [pd.Series([resume]), data_jd],
        ignore_index=True
    )
    tfidf = TfidfVectorizer(stop_words='english',
                            tokenizer=stem_tokenizer,
                            lowercase=True,
                            max_df=0.7,
                            min_df=1,
                            ngram_range=(1, 2)
                           ).fit(data_jd)
  
    description_matrix = tfidf.transform(jobs_list)
    similarity_matrix = linear_kernel(description_matrix)

    job_index = 0

    similarity_score = list(enumerate(similarity_matrix[job_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:item_count + 1]

    job_indices = [i[0] for i in similarity_score]
    return data.iloc[job_indices]

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text
  
data = load_data()
data_jd = data['job_summary']
resume = ''


with st.container():
    col1, col2, col3 = st.columns((2, 0.5, 2))

    with col1:
        uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")
      
        if uploaded_file is not None:
            st.write("PDF Uploaded Successfully")
            resume = input_pdf_text(uploaded_file)

    with col3:
        result_count = st.number_input('Results count', 1, 100, 30)
        st.write('')

if resume != '':
    results = recommend_jobs(resume, result_count)

    with st.container():
        for index, result in results.iterrows():
            with st.expander(result['job_title']):
                st.write('**Location:** ' + result['job_location'])
                st.write('**Company:** ' + result['company'])

                st.markdown('**Job Description**')
                st.write(result['job_summary'])

                st.write(f'**Link:** [{result["job_link"]}]({result["job_link"]})')
