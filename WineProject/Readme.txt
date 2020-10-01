I. File list
------------------

winemag-data-130k-v2_OpenRefine.csv          implementation of group provinces and wine varieties using Open Refine (raw file winemag-data-130k-v2.csv)
 

II. Requirements
------------------
Code can run as is on JupyterNotebook or any other Python editor. 
I used Anaconda distributor to create and run this book. 

Libraries/packages used in the code :
 
import pandas as pd
import matplotlib.pyplot as plt
import squarify 
import numpy as np 
import seaborn as sns
import nltk
import os
import warnings
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re  
from nltk.stem import WordNetLemmatizer 
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

III. Design
--------------------------------------------------
Initial len(dataset) 119988 
1. Clean up duplicates / nulls / 
2. Impute mean of the price by country to fill the null prices 
3. General Visualizations
	TreeMap ( reviews per country)
	Reviews by Scores ( and score buckets )
4.Outlier Analysis ( Price )
5.Variety Analysis and grouping into two labels 
	Price Analysis for two labels
6.Text Mining 
	Define Functions :
		preprocess
		tokenization
		lemmatizer
        	frequent Words
        Visualizations
7.Prepare text for model
	LabelEncoder
	HotEncoder 
8.Create final dataset 
	Merge features from text to rest of features
	Remove extra columns 
9.Model Implementation
	split into train/test
        Apply Random Forest (*) 

Final len(dataset) 104454


* I had many memory issues with my implementation of Random Forest ( from Scratch ). I tried other algorithms as well but my RAM memory always failed with many shutdowns.
After a lot of debugging, my laptop was diagnosed with some memory fail and I run out of time to fix it. 
I decided to provide an scikit learn implementation to conclude the analysis. 

