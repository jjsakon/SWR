#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:29:30 2017

@author: shait
"""

import gensim
import numpy as np


def get_model(word2vecpath):
    #returns a pretrained google word2vec model with the input of the 
    #path where you checked out this repo. Use this model to create a 
    #w2v.mat file (function below) or two get the similarity of any two
    #words with model.similarity(word1, word2). Note that it is 
    #case sensitive
    
    return gensim.models.KeyedVectors.load_word2vec_format(
        word2vecpath+'/GoogleNews-vectors-negative300.bin', 
        binary=True)


def create_word2vec_sim(word2vecpath, model, savepath=None):
    #Input the path where you checked out this repo, 
    #a pretrained word2vec model, and an optional path to save
    #the matrix. This function finds similarity values for all
    #words in the ltpFR wordpool and stores them in a two 
    #dimensional matrix
    
    wordpool = np.loadtxt('Desktop/word2vec/wordpool.txt'
                          , dtype=str)
    
    for i in range(len(wordpool)):
        wordpool[i] = wordpool[i].lower()
        if wordpool[i] == 'doughnut':
            wordpool[i] = 'donut'
    
    sim_mat = np.zeros([len(wordpool), len(wordpool)])
    
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            sim_mat[i, j] = model.similarity(wordpool[i], wordpool[j])
    
    if savepath:
        import scipy.io as sio
        sio.savemat(savepath+'w2v.mat', sim_mat)
    
    return sim_mat


def get_sim(model, word1, word2):
    return model.similarity(word1.lower(), word2.lower())


    