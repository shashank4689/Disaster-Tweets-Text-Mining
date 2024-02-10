#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px

import warnings 
warnings.filterwarnings('ignore')


# In[7]:


df_train = pd.read_csv("C:/Users/shashank/Desktop/lumen/kaggle/twitter/train.csv")


# In[8]:


df_test = pd.read_csv("C:/Users/shashank/Desktop/lumen/kaggle/twitter/test.csv")


# In[11]:


df_train.head()
df_test.head()


# In[12]:


df_test.nunique()


# In[13]:


df_train.nunique()


# In[14]:


df_train.isnull().sum()


# In[15]:


df_test.isnull().sum()


# In[16]:


df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(8, 72), dpi=100)

sns.countplot(y=df_train.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=df_train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

df_train.drop(columns=['target_mean'], inplace=True)


# In[17]:


df_train['text'].apply(lambda x:x.lower())


# In[18]:


df_test['text'].apply(lambda x:x.lower())


# In[24]:


#import library and create a for loop for punctuation removal
import string
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# In[25]:


df_train['text'].apply(remove_punctuations)


# In[26]:


df_test['text'].apply(remove_punctuations)


# In[28]:


import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset if not already downloaded
nltk.download('stopwords')

# Retrieve English stopwords
stop = stopwords.words('english')


# In[29]:


df_train['text_without_stopwords'] = df_train['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))


# In[30]:


df_train['text'] = df_train['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))


# In[33]:


import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')
# Download WordNet corpus if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to perform lemmatization on a single text
def lemmatize_text(text):
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    # Lemmatize each word using WordNet
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]  # Lemmatize verbs
    lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in lemmatized_words]  # Lemmatize nouns
    # Join the lemmatized words back into a single string
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Lemmatize text in 'text' column of df_train
df_train['text'] = df_train['text'].apply(lemmatize_text)

# Lemmatize text in 'text' column of df_test
df_test['text'] = df_test['text'].apply(lemmatize_text)


# In[35]:


#white space removal
df_train['text'] = df_train['text'].apply(lambda x: ' '.join(x.split()))
df_test['text'] = df_test['text'].apply(lambda x: ' '.join(x.split()))


# In[38]:


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Clean text in the 'text' column of df_train
df_train['text'] = df_train['text'].apply(clean_text)

# Clean text in the 'text' column of df_test
df_test['text'] = df_test['text'].apply(clean_text)


# In[39]:


df_train.head()


# In[40]:


df_train.drop(columns=['cleaned_text'], inplace=True)

# Drop 'cleaned_text' column from df_test
df_test.drop(columns=['cleaned_text'], inplace=True)


# In[42]:


df_train.head()


# In[45]:


df_test.head()


# In[47]:


# Calculate the number of missing values in 'keyword' and 'location' columns for target=1 in df_train
missing_values_target1_train = df_train[df_train['target'] == 1][['keyword', 'location']].isnull().sum()

# Calculate the number of missing values in 'keyword' and 'location' columns for target=0 in df_train
missing_values_target0_train = df_train[df_train['target'] == 0][['keyword', 'location']].isnull().sum()



print("Missing values for target=1 in df_train:")
print(missing_values_target1_train)

print("\nMissing values for target=0 in df_train:")
print(missing_values_target0_train)



# In[48]:


import pandas as pd

# Fill missing values in 'keyword' and 'location' columns with a placeholder value, such as 'unknown'
placeholder_value = 'unknown'

# Treat missing values in df_train
df_train['keyword'].fillna(placeholder_value, inplace=True)
df_train['location'].fillna(placeholder_value, inplace=True)

# Treat missing values in df_test
df_test['keyword'].fillna(placeholder_value, inplace=True)
df_test['location'].fillna(placeholder_value, inplace=True)


# In[49]:


df_test.head()


# In[50]:


y = df_train['target']
X = df_train.drop(columns=['target', 'id'])


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[52]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer with max_features=3000
tv = TfidfVectorizer(max_features=3000)


# In[53]:


tv.fit(X_train['text'])


# In[54]:


X_train_cv = tv.transform(X_train['text']).toarray()
X_val_cv = tv.transform(X_val['text']).toarray()


# In[56]:


print(type(X_train))
print(type(X_train_cv))


# In[58]:


train_with_cv = pd.DataFrame(X_train_cv, columns= tv.get_feature_names_out())
train_with_cv.head()


# In[59]:


#import libraries and fit model 

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB().fit(X_train_cv,y_train)


# In[60]:


y_train_pred = nb.predict(X_train_cv)
y_val_pred = nb.predict(X_val_cv)


# In[62]:


from sklearn.metrics import classification_report

print('Train Report ---')
print(classification_report(y_train, y_train_pred))


# In[63]:


print('Validation Report ---')
print(classification_report(y_val, y_val_pred))


# In[64]:


X_test_cv = tv.transform(df_test['text']).toarray()


# In[69]:


#import count_vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)


# In[70]:


cv.fit(X_train['text'])


# In[71]:


X_train_cv = cv.transform(X_train['text']).toarray()
X_val_cv = cv.transform(X_val['text']).toarray()


# In[72]:


print(type(X_train))
print(type(X_train_cv))


# In[73]:


train_with_cv = pd.DataFrame(X_train_cv, columns= cv.get_feature_names_out())
train_with_cv.head()


# In[74]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB().fit(X_train_cv,y_train)
y_train_pred = nb.predict(X_train_cv)
y_val_pred = nb.predict(X_val_cv)


# In[75]:


print('Train Report ---')
print(classification_report(y_train, y_train_pred))


# In[76]:


print('Validation Report ---')
print(classification_report(y_val, y_val_pred))


# In[77]:


X_test_cv = tv.transform(df_test['text']).toarray()


# In[78]:


y_test_pred = nb.predict(X_test_cv)


# In[79]:


print(y_test_pred)
print(y_test_pred.dtype)


# In[80]:


df = pd.DataFrame(y_test_pred)


# In[83]:


df


# In[82]:


sample_submission = pd.read_csv("C:/Users/shashank/Desktop/lumen/kaggle/twitter/sample_submission.csv")


# In[84]:


sample_submission['target'] = df


# In[85]:


sample_submission.to_csv('C:/Users/shashank/Desktop/lumen/kaggle/twitter/sample_submission2.csv', index=False)


# In[ ]:




