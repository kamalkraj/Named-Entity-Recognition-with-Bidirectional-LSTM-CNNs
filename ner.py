import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

class Parser:

    def __init__(self):
        # ::Hard coded char lookup ::
        self.char2Idx = {"PADDING":0, "UNKNOWN":1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            self.char2Idx[c] = len(self.char2Idx)
        # :: Hard coded case lookup ::
        self.case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}

    def load_models(self, loc=None):
        if not loc:
            loc = os.path.join(os.path.expanduser('~'), '.ner_model')
        self.model = load_model(os.path.join(loc,"model.h5"))
        # loading word2Idx
        self.word2Idx = np.load(os.path.join(loc,"word2Idx.npy")).item()
        # loading idx2Label
        self.idx2Label = np.load(os.path.join(loc,"idx2Label.npy")).item()

    def getCasing(self,word, caseLookup):   
        casing = 'other'
        
        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1
                
        digitFraction = numDigits / float(len(word))
        
        if word.isdigit(): #Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif word.islower(): #All lower case
            casing = 'allLower'
        elif word.isupper(): #All upper case
            casing = 'allUpper'
        elif word[0].isupper(): #is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'  
        return caseLookup[casing]

    def createTensor(self,sentence, word2Idx,case2Idx,char2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']
    
        wordIndices = []    
        caseIndices = []
        charIndices = []
            
        for word,char in sentence:  
            word = str(word)
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
            charIdx = []
            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN'])   
            wordIndices.append(wordIdx)
            caseIndices.append(self.getCasing(word, case2Idx))
            charIndices.append(charIdx)
            
        return [wordIndices, caseIndices, charIndices]

    def addCharInformation(self, sentence):
        return [[word, list(str(word))] for word in sentence]

    def padding(self,Sentence):
        Sentence[2] = pad_sequences(Sentence[2],52,padding='post')
        return Sentence

    def predict(self,Sentence):
        Sentence = words =  word_tokenize(Sentence)
        Sentence = self.addCharInformation(Sentence)
        Sentence = self.padding(self.createTensor(Sentence,self.word2Idx,self.case2Idx,self.char2Idx))
        tokens, casing,char = Sentence
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = self.model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1)
        pred = [self.idx2Label[x].strip() for x in pred]
        return list(zip(words,pred))