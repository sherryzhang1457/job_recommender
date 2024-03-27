import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json
from chromadb.config import Settings
import requests

url = "https://jsearch.p.rapidapi.com/search"

querystring_ds = {"query":"Data scientist","page":"1","num_pages":"10","date_posted":"today","employment_types":"FULLTIME, CONTRACTOR","exclude_job_publishers":"Dice, jooble, Clearance Jobs"}
querystring_ml = {"query":"Machine learning","page":"1","num_pages":"10","date_posted":"today","employment_types":"FULLTIME, CONTRACTOR","exclude_job_publishers":"Dice, jooble, Clearance Jobs"}

# "date_posted":"3days" "today"
headers = {
	"X-RapidAPI-Key": "e033f36acdmsh5501b64fe6a088fp1ca60bjsn3c0eacac535f",
	"X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring_ds)
response_ml = requests.get(url, headers=headers, params=querystring_ml)

job_postings_json = response.json()
job_postings_json_ml = response_ml.json()

file_path = "data-ds.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json['data'], f)
file_path = "data-ml.json"
with open(file_path, "w") as f:
    json.dump(job_postings_json_ml['data'], f)

df_ds = pd.read_json('data-ds.json', encoding = 'utf-8')
df_ml = pd.read_json('data-ml.json', encoding = 'utf-8')
df = pd.concat([df_ds, df_ml], axis=0)

df_select = df[
      ['job_title', 'employer_name', 'employer_logo', 'employer_website',
       'employer_company_type', 'job_publisher', 'job_employment_type',
       'job_apply_link', 'job_description',
       'job_is_remote', 'job_city', 'job_state',
       'job_latitude', 'job_longitude', 'job_benefits',
       'job_required_experience', 'job_required_skills',
       'job_required_education', 'job_experience_in_place_of_education',
       'job_highlights']
].copy()


def clean_job_postings(df):
  df['job_location'] = df['job_city'] + ', ' + df['job_state']
  df['info'] = df['job_title'] + '|' + df['job_location'] + '|' + df['employer_name']
  df = df.drop_duplicates(subset='info', ignore_index=True)
  df = df.drop_duplicates(subset='job_description', ignore_index=True)
  df_exp = pd.json_normalize(df['job_required_experience'])
  # df = pd.concat([df.drop(columns=['job_required_experience']), df_exp], axis=1)
  df = pd.concat([df, df_exp], axis=1)
  df['required_experience'] = df['required_experience_in_months'] / 12
  df['required_experience'] = df['required_experience'].fillna(0.0)
  df['citizenship'] = df['job_description'].str.contains('clearance') | (df['job_description'].str.contains('SCI')) | (df['job_description'].str.contains('US citizenship'))| (df['job_description'].str.contains('Clearance')) | (df['job_description'].str.contains('US Citizen'))
  return df

df_select = clean_job_postings(df_select)
# Set up the DataFrame
job_postings = df_select
job_postings = job_postings.dropna(subset=['info', 'job_description'])
job_postings = job_postings.fillna('')

# update vector database from DataFrame
def update_chroma_db(df, collection):

  for index, row in df.iterrows():
    collection.add(
      documents=row['job_description'],
      metadatas=[{"info": row['info'],
                  "minimum": row['required_experience'],
                  "citizen": row['citizenship'],
                  "link": row['job_apply_link']
                 }],
      ids=str(index)
    )
  return collection



# Get chromaDB client
setting = Settings(allow_reset = True)
chroma_client = chromadb.PersistentClient(path='database/', settings = setting)

# chroma_client.delete_collection(name="job_postings")
chroma_client.reset()
collection = chroma_client.create_collection(
        name="job_postings",
        metadata={"hnsw:space": "cosine"}
    )

update_chroma_db(job_postings, collection)

