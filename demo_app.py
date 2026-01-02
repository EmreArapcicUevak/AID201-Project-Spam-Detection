import datetime

import streamlit as st
import kagglehub, os, joblib

@st.cache_resource(ttl=datetime.timedelta(hours=1))
def load_model():
    path = kagglehub.model_download("emrearapcicuevak/sms-spam-detection/scikitLearn/default")
    model_path = os.path.join(path, "Spam_Detection.joblib")
    return joblib.load(model_path)

@st.cache_data(max_entries=5000, ttl=datetime.timedelta(minutes=20))
def predict_spam(sms_message : str) -> bool:
    model = load_model()
    return model.predict([sms_message])[0]

st.title("Spam Detection")
st.markdown("""
This is a demonstration of the SMS Spam Detection model I made for my final project for `Programming for Data Science` ([AID201](https://ecampus.ius.edu.ba/syllabus/aid201-programming-data-science)) at [International
university of Sarajevo](https://www.ius.edu.ba/en)
- The dataset used for this project can be found [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- The source code can be found on my [github](https://github.com/EmreArapcicUevak/AID201-Project-Spam-Detection)
---
""")

query = st.text_input(label="SMS Message", value=None, placeholder="Enter your sms message", icon='✉️')
if query:
    with st.spinner('Running...', show_time=True):
        result = predict_spam(query)

    if result:
        st.error( 'This message is SPAM!!!', icon="🚨" )
    else:
        st.success( 'This is HAM', icon="✅" )
