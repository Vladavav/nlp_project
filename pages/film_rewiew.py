import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

st.title('Классификация отзыва на фильм')
st.write('Введите текст для анализа')
uploaded_text = st.text_input('Введите текст')

label_dict = {1: 'Good', 0: 'Bad', 2: 'Neutral'}

if uploaded_text:
    tokenizer_tiny2 = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model_tiny2 = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    # Токенизация введенного текста
    encoded_review = tokenizer_tiny2(uploaded_text, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    # st.write(encoded_rewies)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_tiny2 = model_tiny2.to(device)
    encoded_review = {key: value.to(device) for key, value in encoded_review.items()}
    # Получение векторного представления отзыва
    with torch.no_grad():
        model_out = model_tiny2(**encoded_review)
        vector = model_out.last_hidden_state[:, 0, :].cpu().numpy()
    # Загрузка LogReg 
    bert_log_reg = joblib.load('models/bert_log_reg.pkl')
    predict_log_reg = bert_log_reg.predict(vector)
    predict_label = label_dict[predict_log_reg[0]]
    st.write(f'Предсказанная категория отзыва - {predict_label}')



