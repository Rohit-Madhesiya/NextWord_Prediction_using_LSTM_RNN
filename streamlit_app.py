import streamlit as st 
import numpy as np 
import pickle
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# load the LSTM model
model=load_model('next_word_lstm.h5')

# load the Tokenizer
with open('tokenizer.pkl','rb') as file:
  tokenizer=pickle.load(file)

# function to predict the next word
def predict_next_word(model,tokenier,text,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if(len(token_list)>=max_sequence_len):
    token_list=token_list[-(max_sequence_len-1):] #ensures the sequence length matches max_sequence_len-1
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None

# Streamlit App
st.title('Next Word Prediction with LSTM RNN')
input_text=st.text_input('Enter the Sequence of Words',"To be or not to")
if st.button("Predict Next Word"):
  max_sequence_len=model.input_shape[1]+1 #Retrieve the max sequence length from the max_sequence_len-1
  next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
  st.write(f'Next Word: {next_word}')

