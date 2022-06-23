# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:04:38 2022

This script is a modules for text_segmentation project

@author: Afiq Sabqi
"""

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.layers import Bidirectional,Embedding

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_tens_layer(self,max_len=340,vocab_size=30000,
                          embedding_dim=128,num_nodes=128,
                          drop_rate=0.3):
        model=Sequential()
        model.add(Input(shape=(max_len)))
        model.add(Embedding(vocab_size,embedding_dim))
        model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_nodes))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_nodes,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(5,'softmax'))
        model.summary()
        return model