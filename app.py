import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Load data from CSV
@st.cache_data
def load_context_from_csv():
    df = pd.read_csv("data.csv")
    paragraphs = []
    for _, row in df.iterrows():
        paragraph = f"فرع {row['اسم الفرع']} يقع في {row['المدينة']}. أوقات العمل: {row['أوقات الدوام']}. رضا العملاء: {row['رضا العملاء']}. رقم المدير: {row['رقم المدير']}."
        paragraphs.append(paragraph)
    return "\n".join(paragraphs)

context = load_context_from_csv()

# Title
st.title("🤖 شات بوت فروع Brew & Bite")

# Input
user_question = st.text_input("📝 اكتب سؤالك:")

# Function to generate answer
def generate_answer(question, context):
    prompt = f"استخدم المعلومات التالية للإجابة: {context}\nالسؤال: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Output
if user_question:
    answer = generate_answer(user_question, context)
    st.success("🤖 الرد: " + answer)
