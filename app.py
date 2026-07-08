# importing reruire libraries
import streamlit as st 
import pickle
import numpy as np


# read the two pickle file
model=pickle.load(open('model.pkl','rb'))
tv=pickle.load(open('tv_transform.pkl','rb'))


### MAIN FUNCTION ###
def main(title = "AI-Powered System".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 25px; color: blue;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    info = ''
    
    with st.expander("Type your text here"):
        text_message = st.text_input("Please enter your message")
        if st.button("Predict"):
            prediction = model.predict(tv.transform([text_message]).toarray())

            if(prediction[0] == 0):
                info = 'Neutral'

            elif(prediction[0] == 1):
                info= 'Positive'

            else:
                info = 'Negative'
            st.success('Response: {} feedback'.format(info))

if __name__ == "__main__":
    main()
