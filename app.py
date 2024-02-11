from dotenv import load_dotenv
import json
import os
import PyPDF2 as pdf
import streamlit as st
import google.generativeai as genai

# use gemini pro LLM model API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

