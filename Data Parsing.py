# =============================================================================
# importing required dictionaries
# =============================================================================

import pandas as pd
import numpy as np
import os
import glob
import re
import nltk
from nltk.corpus import stopwords   
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from operator import itemgetter

# =============================================================================
# Setting up working direcotry and loading files
# =============================================================================
os.getcwd()
os.chdir("D:\\Data Science Project\\Conversation Mining")
files_path = "D:\\Data Science Project\\Conversation Mining\\Chat Transcripts for Project"
files_pathe_list = glob.glob(os.path.join(files_path,"*.text"))

# reading all 91 files and saving in FullData.test in the working directory
with open("full_data.text", "wb") as outfile:
    for f in files_pathe_list:
        with open(f, "rb") as infile:
            outfile.write(infile.read())

#Reading text file as FullData            
data = open("full_data.text", 'r',encoding='utf-8')
FullData = data.readlines()  
print(FullData[0:21])
data.close()


df1 = pd.DataFrame (FullData,columns=['V1'])
df2 = df1 #keeping df1 original compiled datafeame as backup and woring on df2
df2 = df2.replace(r'^\s*$', np.NaN, regex=True) #Replacing blank values (white space) with NaN
df2 = df2.dropna() #dropping nan values
df2 = df2.reset_index(drop=True) #reset index

# =============================================================================
# Separating structured 14 variables
# =============================================================================
len(df2[df2['V1'].str.contains("City:")])
len(df2[df2['V1'].str.contains("Unread:")])
# as number of observations in City variable and Unread variable differ, we can not column bind all 14 variable by taking each variable out separately

#running for loop for entire dataset column and collecting first 14 rows from chat separator which contains structured 14 variables observation
df3 = pd.DataFrame( df2.iloc[0:14,0])

for j in range(0,2301764):
    if df2.iloc[j,0] == df2.iloc[20,0]:
        for i in range(1,15):
            k=df2.iloc[j+i,]
            df3 = df3.append(k,ignore_index=True)

df3.to_csv('Python_df3.csv', index = False) #saving file as Python_df3.csv in the woring directory

#converting separated 14 variables in one column into rows
df3 = pd.read_csv("Python_df3.csv")
df4 = pd.DataFrame()
k = pd.DataFrame()
j=0
while (j < 1789536):
    for i in range(0,14):
            k[i,]= df3.iloc[i+j,]
    df4 = df4.append(k,ignore_index=True)
    j=j+14

#setting up column names
df4.columns = ['Timestamp','Unread','Visitor_ID','Visitor_Name','Visitor_Email','Visitor_Notes','IP','Country_Code','Country_Name','Region','City','User_Agent','Platform','Browser']
# removing column names from rows
df4['Timestamp'] = df4['Timestamp'].str.replace('Timestamp:', '')
df4['Timestamp'] = df4['Timestamp'].str.replace('T', ' ')
df4['Timestamp'] = df4['Timestamp'].str.replace('Z', '')
df4['Timestamp'] = pd.to_datetime(df4['Timestamp'])
df4['Unread'] = df4['Unread'].str.replace('Unread:', '', regex=True)
df4['Visitor_ID'] = df4['Visitor_ID'].str.replace('Visitor ID:', '', regex=True)
df4['Visitor_Name'] = df4['Visitor_Name'].str.replace('Visitor Name:', '', regex=True)
df4['Visitor_Email'] = df4['Visitor_Email'].str.replace('Visitor Email:', '', regex=True)
df4['Visitor_Notes'] = df4['Visitor_Notes'].str.replace('Visitor Notes:', '', regex=True)
df4['IP'] = df4['IP'].str.replace('IP:', '', regex=True)
df4['Country_Code'] = df4['Country_Code'].str.replace('Country Code:', '', regex=True)
df4['Country_Name'] = df4['Country_Name'].str.replace('Country Name:', '', regex=True)
df4['Region'] = df4['Region'].str.replace('Region:', '', regex=True)
df4['City'] = df4['City'].str.replace('City:', '', regex=True)
df4['User_Agent'] = df4['User_Agent'].str.replace('User Agent:', '', regex=True)
df4['Platform'] = df4['Platform'].str.replace('Platform:', '', regex=True)
df4['Browser'] = df4['Browser'].str.replace('Browser:', '', regex=True)
#saving structurd dataframe as Structured_Python.csv in the working directory 
df4.to_csv('Structured_Python.csv', index = False)

# =============================================================================
# Further analysis of structured data shall be conducted on Tableau
# =============================================================================


# =============================================================================
# ngrams and topic modeling
# =============================================================================

df5 = df2 # lodading full data into df5
df5.iloc[0:14,0]=float('NaN') #replacing 14 structured variables by NaN and removing
for j in range(0,2301764):
    if df5.iloc[j,0] == df5.iloc[20,0]:
        for i in range(1,15):
            df5.iloc[j+i,]=float('NaN')
df5 = df5.dropna()
df5 = df5.reset_index(drop=True)

df5.to_csv('Chattranscripts_Python.csv', index = False) #saving chat between visitor and bot as Chattranscripts_Python.csv

# =============================================================================
# Ngrams on chats between visitor and bot
# =============================================================================
#reading chat transcript file and cleaning
df5 = pd.read_csv("Chattranscripts_Python.csv")
#removing chat seperator and removing opening statements of bots (mounica patel, ananya and customer service)
df5 = df5[~(df5['V1'].str.contains("================================================================================\n"))]
df5 = df5[~(df5['V1'].str.contains("Thank you for connecting with ExcelR!"))]
df5 = df5[~(df5['V1'].str.contains("Hello, how may I be of assistance to you?"))]
df5 = df5[~(df5['V1'].str.contains("Hi Welcome to ExcelR Solutions. How can i assist you?"))]

# Remove punctuation
df5['V1'] = df5['V1'].map(lambda x: re.sub('[,"-\.()!@:-?]', '', x))
# Convert the titles to lowercase
df5['V1'] = df5['V1'].map(lambda x: x.lower())
# Remove numbers
df5['V1'] = df5['V1'].str.replace('\d+', '')

df5 = df5.stack().str.strip().unstack()
#inspect first 10 rows
df5.iloc[0:10,0]

#convert cleaned dataframe to list
lst = df5['V1'].tolist()

#preparing stopwords
def prepareStopWords():
    stopwordsList = []
    stopwordsList = stopwords.words('english')
    stopwordsList.append('mounica')
    stopwordsList.append('patel')
    stopwordsList.append('ananya')
    stopwordsList.append('visitor')
    return stopwordsList

#joining all strings into one string
rawText  = " ".join(lst)

#tokenize rawtext words
tokens = nltk.word_tokenize(rawText)
text = nltk.Text(tokens)

# Remove extra chars and remove stop words.
stopWords = prepareStopWords()
text_content = [word for word in text if word not in stopWords]
# Remove any entries where the len is zero.
text_content = [s for s in text_content if len(s) != 0]
#get the lemmas of each word to reduce the number of similar words
WNL = nltk.WordNetLemmatizer()
text_content = [WNL.lemmatize(t) for t in text_content]

# =============================================================================
# UNIGRAM
# =============================================================================

unigram_strg  = " ".join(text_content)

# Setting word cloud params and plotting the word cloud.
WC_height = 500
WC_width = 1000
WC_max_words = 100

unigram_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(unigram_strg)
plt.figure()
plt.imshow(unigram_wordcloud, interpolation="bilinear")
plt.title('Most frequently occurring words in bots and visitors chat')
plt.axis("off")
plt.show()


# =============================================================================
# BIGRAM
# =============================================================================
    
# setup and score the bigrams using the raw frequency.
bigram_finder = BigramCollocationFinder.from_words(text_content)
bigram_measures = BigramAssocMeasures()
scored = bigram_finder.score_ngrams(bigram_measures.raw_freq)

# Sort highest to lowest based on the score.
scoredList = sorted(scored, key=itemgetter(1), reverse=True)

word_dict = {}
listLen = len(scoredList)

# Get the bigram and make a contiguous string for the dictionary key. Set the key to the scored value. 
for i in range(listLen):
    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

bigram_wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_dict)
plt.figure()
plt.imshow(bigram_wordCloud, interpolation="bilinear")
plt.title('Most frequently occurring bigrams in bots and visitors chat are connected with an underscore')
plt.axis("off")
plt.show()

# =============================================================================
# Trigram
# =============================================================================


trigram_finder = TrigramCollocationFinder.from_words(text_content)
trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio,10)

trigram_measures = TrigramAssocMeasures()
scored = trigram_finder.score_ngrams(trigram_measures.raw_freq)

# Sort highest to lowest based on the score.
scoredList = sorted(scored, key=itemgetter(1), reverse=True)

word_dict = {}
listLen = len(scoredList)
 
for i in range(listLen):
    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

Trigram_wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_dict)
plt.figure()
plt.imshow(Trigram_wordCloud, interpolation="bilinear")
plt.title('Most frequently occurring trigrams in bots and visitors chat are connected with an underscore')
plt.axis("off")
plt.show()


# =============================================================================
# Separating only Visitors chat and plotting Unigram, Bigram and Trigram
# =============================================================================
#separating visitors chat in df76
df6 = df5[df5['V1'].str.contains("visitor")]

#convert cleaned dataframe to list
lst = df6['V1'].tolist()

#joining all strings into one string
rawText  = " ".join(lst)
#tokenize rawtext words
tokens = nltk.word_tokenize(rawText)
text = nltk.Text(tokens)

def prepareStopWords():
    stopwordsList = []
    stopwordsList = stopwords.words('english')
    stopwordsList.append('ok')
    stopwordsList.append('mounica')
    stopwordsList.append('patel')
    stopwordsList.append('ananya')
    stopwordsList.append('visitor')
    stopwordsList.append('hi')
    stopwordsList.append('get')
    stopwordsList.append('end')
    return stopwordsList

stopWords = prepareStopWords()
text_content = [word for word in text if word not in stopWords]
# Remove any entries where the len is zero.
text_content = [s for s in text_content if len(s) != 0]
#get the lemmas of each word to reduce the number of similar words
WNL = nltk.WordNetLemmatizer()
text_content = [WNL.lemmatize(t) for t in text_content]


# =============================================================================
# UNIGRAM
# =============================================================================

unigram_strg  = " ".join(text_content)

WC_height = 500
WC_width = 1000
WC_max_words = 100

unigram_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(unigram_strg)
plt.figure()
plt.imshow(unigram_wordcloud, interpolation="bilinear")
plt.title('Most frequently occurring words in only visitors chat')
plt.axis("off")
plt.show()

# =============================================================================
# BIGRAM
# =============================================================================
    
bigram_finder = BigramCollocationFinder.from_words(text_content)
bigram_measures = BigramAssocMeasures()
scored = bigram_finder.score_ngrams(bigram_measures.raw_freq)

scoredList = sorted(scored, key=itemgetter(1), reverse=True)

word_dict = {}
listLen = len(scoredList)

for i in range(listLen):
    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

bigram_wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_dict)
plt.figure()
plt.imshow(bigram_wordCloud, interpolation="bilinear")
plt.title('Most frequently occurring bigrams in visitors chat are connected with an underscore')
plt.axis("off")
plt.show()

# =============================================================================
# Trigram
# =============================================================================

trigram_finder = TrigramCollocationFinder.from_words(text_content)
trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio,10)

trigram_measures = TrigramAssocMeasures()
scored = trigram_finder.score_ngrams(trigram_measures.raw_freq)

scoredList = sorted(scored, key=itemgetter(1), reverse=True)

word_dict = {}
listLen = len(scoredList)

for i in range(listLen):
    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

Trigram_wordCloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_dict)
plt.figure()
plt.imshow(Trigram_wordCloud, interpolation="bilinear")
plt.title('Most frequently occurring trigrams in visitors chat are connected with an underscore')
plt.axis("off")
plt.show()