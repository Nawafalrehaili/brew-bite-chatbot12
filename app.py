import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# Load data (ensure the CSV is in the same directory or use an absolute path)
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# Create a mapping of city to paragraph
def get_context(city):
    row = df[df['city'].str.contains(city, case=False, na=False)]
    if not row.empty:
        return row.iloc[0]['context']
    return ""

# Generate answer
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🤖 شات بوت فروع Brew & Bite")

user_question = st.text_input("📝 اكتب سؤالك:")
if user_question:
    city_found = False
    for city in df['city']:
        if city in user_question:
            context = get_context(city)
            answer = generate_answer(user_question, context)
            st.success(f"💬 الرد: {answer}")
            city_found = True
            break

    if not city_found:
        st.warning("⚠️ لم يتم العثور على مدينة في سؤالك. تأكد من ذكر اسم المدينة بشكل صحيح.")
