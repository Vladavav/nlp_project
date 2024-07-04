import streamlit as st
import torch


st.title('Классификация отзыва на фильм')
st.write('Введите текст для анализа')
uploaded_text = st.text_input('Введите текст')