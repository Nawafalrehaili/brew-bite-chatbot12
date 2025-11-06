import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def generate_answer(question):
    tokenizer, model = load_model()
    input_text = f"Answer the following question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("🤖 شات بوت فروع Brew & Bite")
user_question = st.text_input("✏️ اكتب سؤالك:")

if user_question:
    answer = generate_answer(user_question)
    st.success(f"✅ الرد: {answer}")
