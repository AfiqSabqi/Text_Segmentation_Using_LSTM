# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:24:08 2022

This text is to build a model for text segmentation analysis

source credit to :
    https://raw.githubusercontent.com/susanli2016/
    PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

@author: Afiq Sabqi
"""

import re
import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix

from modules_for_text_segmentation import ModelCreation

#%%                                 STATIC
CSV_URL='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

log_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH=os.path.join(os.getcwd(),'logs',log_dir)

MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')
TOKENIZER_PATH=os.path.join(os.getcwd(),'model','tokenizer_sentiment.json')
OHE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')

#%%                               DATA LOADING

df=pd.read_csv(CSV_URL)


#%%                              DATA INSPECTION


df.head(10)
df.info()
df.duplicated().sum()
df[df.duplicated()]
# although it show have 99 duplicated. should not be drop because what is 
# said to be duplicated which is not. all are different
df['category'][0]
df['text'][0]


#%%                               DATA CLEANING

category=df['category'].values 
text=df['text'].values 


# in case there is a funny characters in text and capital letter
for index,tex in enumerate(text):
    text[index]=re.sub('<.*?>',' ',tex)
    text[index]=re.sub('[^a-zA-Z]',' ',tex).lower().split()



#%%                            FEATURES SELECTION

pass 
# because it is all text, nothing to select


#%%                              PREPROCESSING

##            Tokenization

vocab_size=30000
oov_token='OOV'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)
word_index=tokenizer.word_index
print(word_index)

# to convert into numbers
train_sequences=tokenizer.texts_to_sequences(text) 

##           Padding and Truncating

# to check the numbers of 'number'
len(train_sequences[1])

# list comprehension
length_of_text=[len(i) for i in train_sequences]

# to get the mean and median number of length for padding
np.mean(length_of_text) 
np.median(length_of_text)

# taking from the median
max_len=340

padded_text=pad_sequences(train_sequences,maxlen=max_len,
                            padding='post',
                            truncating='post')


##          OneHotEncoder for category(target)
ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))


##           Train Test Split
X_train,X_test,y_train,y_test=train_test_split(padded_text,category,
                                               test_size=0.3,
                                               random_state=123)

X_train=np.expand_dims(X_train,axis=-1)
X_test=np.expand_dims(X_test,axis=-1)


#%%                              MODEL DEVELOPMENT

# use LSTM

embedding_dim=128

mc=ModelCreation()
model=mc.simple_tens_layer(max_len)

plot_model(model)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])


#%%                              MODEL EVALUATION

tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

hist=model.fit(X_train,y_train,
               validation_data=(X_test,y_test),
               batch_size=128,epochs=100,
               callbacks=[tensorboard_callback])


hist.history.keys()

training_loss=hist.history['loss']
training_acc=hist.history['acc']
validation_acc=hist.history['val_acc']
validation_loss=hist.history['val_loss']

plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.legend(['train_loss','val_loss'])
plt.show

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)
plt.legend(['train_acc','val_acc'])
plt.show

results=model.evaluate(X_test,y_test)
print(results)

y_true=y_test
y_pred=model.predict(X_test)


y_true=np.argmax(y_true,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

#%%                              MODEL SAVE

model.save(MODEL_SAVE_PATH)

# saving tokenizer
token_json=tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

# saving OneHotEncoder
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)


#%%                          DISCUSSION

'''

    *model accuracy and f1 gives 82% score
    
    *training graph shows an overfitting since the training accuracy is
    higher than validation accuracy
    
    *however since the model accuracy and f1 score gives above 80%, the
    model is consider great and its learning from the training.
    
    *although Earlystopping can overcome the overfitting. in this case
    it seems not give any effect.
    
    *This project has to use LSTM, if not can try other architecture like
    BERT, transformer or GPT3 model.
    
'''




























