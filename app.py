import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    paragraphs = []
    for _, row in df.iterrows():
        paragraph = f"فرع {row['اسم الفرع']} في {row['المدينة']}, أوقات الدوام: {row['أوقات الدوام']}, رضا العملاء: {row['رضا العملاء']}, رقم المدير: {row['رقم المدير']}."
        paragraphs.append(paragraph)
    return "\n".join(paragraphs)

tokenizer, model = load_model()
context = load_data()

def generate_answer(question, context):
    input_text = f"السؤال: {question}\nالمعلومات: {context}\nالجواب:"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🤖 شات بوت فروع Brew & Bite")

user_question = st.text_input("✏️ اكتب سؤالك:")

if user_question:
    answer = generate_answer(user_question, context)
    st.success(f"✅ الرد: {answer}")
