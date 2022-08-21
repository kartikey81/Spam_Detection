import streamlit as st 



st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
from random import sample
from tabnanny import verbose
import pandas as pd # Used for reading the csv data
from nltk.corpus import stopwords
import string # For punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
stopwords = stopwords.words("english")


def review(text):
  df = pd.read_csv("spam.csv", encoding = 'latin-1')

#df.info()

# There are some unwanted extra columns in the data file. To remove them,
  df = df.iloc[:, :2]

  df.columns = ['target', 'message'] # Change column names


# Sets ham to 0, spam to 1
  encoder=LabelEncoder()
  df['target']=encoder.fit_transform(df['target'])

# df['message_length'] = df.message.apply(len)

  

  def data_preparation(message):

    """Removes stopwords and punctuations
    Args:
        message (string): message
    Returns:
        string: new cleaned message
    """
    # messages = df["message"] # Messages column
    punctuations = string.punctuation

    words = []
    for word in message.split():
        word = word.lower()
        if word not in stopwords:
            chars = []
            for char in word:
                if char not in punctuations:
                    chars.append(char)
                else:
                    chars.append(" ")
            
            new_word = "".join(chars)
            words.append(new_word) 
    
    new_message = " ".join(words)
    
    return new_message
    

# Add cleaned_messages to df
  df['cleaned_message'] = df.message.apply(data_preparation)


  targets = df.target
  messages = df.cleaned_message
# print(df.cleaned_message[1084])

# Split train and test data
# - train_test_split -
#   - Split arrays or matrices into random train and test subsets
#   - test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#   - random_state: Controls the shuffling applied to the data before applying the split.
#   - stratify: mmm

  messages_train, messages_test, targets_train, targets_test = train_test_split(messages, targets, test_size=0.2, random_state=20)

# mx = len(max(messages, key=len))

# Tokenize and padding

  num_words = 50000 # The maximum number of words to keep, based on word frequency. 
  max_len = 91

  tokenizer = Tokenizer(num_words = num_words) 
  tokenizer.fit_on_texts(messages_train) # Updates internal vocabulary based on a list of texts.

# Tokenize and paddin for train dataset

  messages_train_features = tokenizer.texts_to_sequences(messages_train) # Updates internal vocabulary based on a list of sequences.
# print(len(max(messages_train_features, key=len))) 79
  messages_train_features = sequence.pad_sequences(messages_train_features, maxlen = max_len)

# Tokenize and paddin for test dataset

  messages_test_features = tokenizer.texts_to_sequences(messages_test)
# print(len(max(messages_test_features, key=len))) #91
  messages_test_features = sequence.pad_sequences(messages_test_features, maxlen = max_len)

  print(len(messages_train_features), len(messages_train_features[0]))
  print(len(messages_test_features), len(messages_test_features[0]))
  from tensorflow import keras
  model = keras.models.load_model('spammodel.h5')
  y_predict  = [1 if o>0.5 else 0 for o in model.predict(messages_test_features)]
  cf_matrix =confusion_matrix(targets_test,y_predict)
  print(cf_matrix)
  sample_texts = ["Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005"        ]
  sample_texts = [data_preparation(sentence) for sentence in sample_texts]

  txts = tokenizer.texts_to_sequences(sample_texts)
  txts = sequence.pad_sequences(txts, maxlen=max_len)
  preds = model.predict(txts, verbose=0)
  print(preds)
  print(np.around(preds))
  pred=np.around(preds)
  print(pred[0][0])

  if pred[0][0]==1.0:
    result= "It is Spam"
  else:
    result="It is not Spam" 

 
    
  return result
html_temp = """
   <div class="" style="background-color:green;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Major Project 2022</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
st.header("Spam Detection System ")
  
  
text = st.text_area("Write Message ")

if st.button("Spam Detection"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
      
if st.button("About"):
  st.subheader("Developed by Kartik Pal")
  st.subheader("Student , Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Major Project 2022 Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
