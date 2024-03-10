# streamlit run main.py

import streamlit as st
import pickle
import time
import cleantext
import pandas as pd

st.title("Twitter Sentiment Analysis")
with st.expander("Predict Text"):
    # Load model
    model = pickle.load(open('twitter_sentiment.pkl', 'rb'))

    # Predict text sentiment
    text = st.text_input("Type here: ")

    submit = st.button("Predict")

    if submit:
        start = time.time()
        prediction = model.predict([text])
        end = time.time()
        st.write("Prediction time taken: ", round(end-start, 4), "seconds")
        print(prediction[0])
        st.write("Predicted Sentiment is: ", prediction[0])

    # Return clean text
    pre_text = st.text_input("Clean Text: ")
    if pre_text:
        st.write(cleantext.clean(pre_text, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))


with st.expander('Predict CSV file'):
    # Load model
    model = pickle.load(open('twitter_sentiment.pkl', 'rb'))

    upl = st.file_uploader('Upload file')

    def sentiment(x):
        a = model.predict(x)
        return a

    if upl:
        df = pd.read_excel(upl)
        df['Predict Sentiment'] = df[['Tweets']].apply(sentiment)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
