---
layout: single
classes: wide
title: "Analyzing gender bias in movie dialogues"
excerpt: "Building a gender classifier model based on the dialogues of characters in Hollywood movies"
date: 2020-11-22
tags:
  - python
  - ml
comments: true
---


In this post, I've tried to analyze gender bias in Hollywood movies using the character dialogues & some movie metadata. The gender bias can be established if we can predict the gender of a Hollywood movie character based on his / her dialogues in the movie. The [dataset](https://www.kaggle.com/Cornell-University/movie-dialog-corpus) was released by [Cornell University](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The data pre-processing, EDA & modeling are all done in Python3 in a Jupyter notebook environment, rendered finally into a Markdown for this blog.

### Import necessary libraries


```python
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from imblearn.under_sampling import RandomUnderSampler
import eli5

import IPython
from IPython.display import display
import graphviz
from sklearn.tree import export_graphviz
import re


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
```

    /opt/conda/lib/python3.7/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.metrics.scorer module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.metrics. Anything that cannot be imported from sklearn.metrics is now part of the private API.
      warnings.warn(message, FutureWarning)
    /opt/conda/lib/python3.7/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.feature_selection.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.feature_selection. Anything that cannot be imported from sklearn.feature_selection is now part of the private API.
      warnings.warn(message, FutureWarning)


### Reading the dataset


```python
lines_df = pd.read_csv('../input/movie_lines.tsv', sep='\t', error_bad_lines=False,
                       warn_bad_lines=False, header=None)
characters_df = pd.read_csv('../input/movie_characters_metadata.tsv', sep='\t', 
                            warn_bad_lines=False, error_bad_lines=False, header=None)

characters_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>u0</td>
      <td>BIANCA</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>f</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>u1</td>
      <td>BRUCE</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>u2</td>
      <td>CAMERON</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>m</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>u3</td>
      <td>CHASTITY</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u4</td>
      <td>JOEY</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>m</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Adding column names to characters dataframe


```python
characters_df.columns=['chId','chName','mId','mName','gender','posCredits']
characters_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chId</th>
      <th>chName</th>
      <th>mId</th>
      <th>mName</th>
      <th>gender</th>
      <th>posCredits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>u0</td>
      <td>BIANCA</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>f</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>u1</td>
      <td>BRUCE</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>u2</td>
      <td>CAMERON</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>m</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>u3</td>
      <td>CHASTITY</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u4</td>
      <td>JOEY</td>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>m</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
characters_df.shape
```




    (9034, 6)



Checking the distribution of gender in the characters dataset


```python
characters_df.gender.value_counts()
```




    ?    6008
    m    1899
    f     921
    M     145
    F      44
    Name: gender, dtype: int64



We need to clean this column. Let's also remove the characters where gender information is not available.
We'll assign a label of 0 to male characters & 1 to female characters.


```python
characters_df = characters_df[characters_df.gender != '?']
characters_df.gender = characters_df.gender.apply(lambda g: 0 if g in ['m', 'M'] else 1)  ## Label encoding

characters_df.shape
```




    (3026, 6)




```python
characters_df.gender.value_counts()
```




    0    2044
    1     982
    Name: gender, dtype: int64



Let's also take a look at the position of the character in the post credits of the movie


```python
characters_df.posCredits.value_counts()
```




    1       497
    2       443
    3       352
    ?       330
    4       268
    5       211
    6       169
    7       125
    8       100
    9        79
    10       54
    11       40
    1000     38
    13       33
    12       32
    16       26
    14       24
    18       24
    17       19
    19       18
    15       14
    21       13
    22        9
    20        8
    29        7
    27        6
    24        5
    25        5
    26        5
    45        4
    23        4
    31        4
    35        4
    38        3
    43        3
    33        3
    34        3
    36        2
    59        2
    39        2
    30        2
    42        2
    28        2
    32        2
    51        1
    82        1
    44        1
    70        1
    46        1
    41        1
    63        1
    37        1
    50        1
    49        1
    47        1
    62        1
    71        1
    Name: posCredits, dtype: int64



The position of characters in the credits section seems to be a useful feature for classification. We can try to use it as a categorical variable later. But let's combine the low frequency ones together first.


```python
characters_df.posCredits = characters_df.posCredits.apply(lambda p: '10+' if not p in ['1', '2', '3', '4', '5', '6', '7', '8', '9'] else p)  ## Label encoding
characters_df.posCredits.value_counts()
```




    10+    782
    1      497
    2      443
    3      352
    4      268
    5      211
    6      169
    7      125
    8      100
    9       79
    Name: posCredits, dtype: int64



Let's clean the lines dataframe now!


```python
lines_df.columns = ['lineId','chId','mId','chName','dialogue']
lines_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lineId</th>
      <th>chId</th>
      <th>mId</th>
      <th>chName</th>
      <th>dialogue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L1045</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>They do not!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L1044</td>
      <td>u2</td>
      <td>m0</td>
      <td>CAMERON</td>
      <td>They do to!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L985</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>I hope so.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L984</td>
      <td>u2</td>
      <td>m0</td>
      <td>CAMERON</td>
      <td>She okay?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L925</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Let's go.</td>
    </tr>
  </tbody>
</table>
</div>



Let's join lines_df and characters_df together.


```python
df = pd.merge(lines_df, characters_df, how='inner', on=['chId','mId', 'chName'],
         left_index=False, right_index=False, sort=True,
         copy=False, indicator=False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lineId</th>
      <th>chId</th>
      <th>mId</th>
      <th>chName</th>
      <th>dialogue</th>
      <th>mName</th>
      <th>gender</th>
      <th>posCredits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L1045</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>They do not!</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L985</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>I hope so.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L925</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Let's go.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L872</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Okay -- you're gonna need to learn how to lie.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L869</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Like my fear of wearing pastels?</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (229309, 8)



Remove empty dialogues from the dataset


```python
df = df[df['dialogue'].notnull()]
df.shape
```




    (229106, 8)



Let's check what kind of movie metadata we can add to our dataset.


```python
movies = pd.read_csv("../input/movie_titles_metadata.tsv", sep='\t', error_bad_lines=False,
                       warn_bad_lines=False, header=None)
movies.columns = ['mId','mName','releaseYear','rating','votes','genres']

movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mId</th>
      <th>mName</th>
      <th>releaseYear</th>
      <th>rating</th>
      <th>votes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>m0</td>
      <td>10 things i hate about you</td>
      <td>1999</td>
      <td>6.9</td>
      <td>62847.0</td>
      <td>['comedy' 'romance']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>m1</td>
      <td>1492: conquest of paradise</td>
      <td>1992</td>
      <td>6.2</td>
      <td>10421.0</td>
      <td>['adventure' 'biography' 'drama' 'history']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>m2</td>
      <td>15 minutes</td>
      <td>2001</td>
      <td>6.1</td>
      <td>25854.0</td>
      <td>['action' 'crime' 'drama' 'thriller']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>m3</td>
      <td>2001: a space odyssey</td>
      <td>1968</td>
      <td>8.4</td>
      <td>163227.0</td>
      <td>['adventure' 'mystery' 'sci-fi']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>m4</td>
      <td>48 hrs.</td>
      <td>1982</td>
      <td>6.9</td>
      <td>22289.0</td>
      <td>['action' 'comedy' 'crime' 'drama' 'thriller']</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_yr = movies[['mId', 'releaseYear']]
movie_yr.releaseYear = pd.to_numeric(movie_yr.releaseYear.apply(lambda y: str(y)[0:4]), errors='coerce')
movie_yr = movie_yr.dropna()
```

We will just add the year of movie release to our dataset. 

```python
df = pd.merge(df, movie_yr, how='inner', on=['mId'],
         left_index=False, right_index=False, sort=True,
         copy=False, indicator=False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lineId</th>
      <th>chId</th>
      <th>mId</th>
      <th>chName</th>
      <th>dialogue</th>
      <th>mName</th>
      <th>gender</th>
      <th>posCredits</th>
      <th>releaseYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L1045</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>They do not!</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L985</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>I hope so.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L925</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Let's go.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L872</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Okay -- you're gonna need to learn how to lie.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L869</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Like my fear of wearing pastels?</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Engineering
- Length of lines
- Count of lines
- One hot encodings for tokens


```python
df['lineLength'] = df.dialogue.str.len()             ## Length of each line by characters
df['wordCountLine'] = df.dialogue.str.count(' ') + 1 ## Length of each line by words
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lineId</th>
      <th>chId</th>
      <th>mId</th>
      <th>chName</th>
      <th>dialogue</th>
      <th>mName</th>
      <th>gender</th>
      <th>posCredits</th>
      <th>releaseYear</th>
      <th>lineLength</th>
      <th>wordCountLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L1045</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>They do not!</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L985</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>I hope so.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L925</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Let's go.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L872</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Okay -- you're gonna need to learn how to lie.</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>46</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L869</td>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>Like my fear of wearing pastels?</td>
      <td>10 things i hate about you</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>32</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Next, let's convert the dialogues into clean tokens 
<ol>
<li>Remove Stopwords : because they occur very often, but serve no meaning. For eg. : is,am,are,the.</li>
<li>Turn all word to smaller cases : I, i -> i</li>
<li>Lemmatization: convert words to their root form. For eg., walk,walks -> walk or geographical,geographic -> geographic</li>
</ol>


```python
wordnet_lemmatizer = WordNetLemmatizer()
def clean_dialogue( dialogue ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # Source : https://www.kaggle.com/akerlyn/wordcloud-based-on-character
    #
    # 1. Remove HTML
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", dialogue) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))   
    
    # 5. Use lemmatization and remove stop words
    meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

df['cleaned_dialogue'] = df['dialogue'].apply(clean_dialogue)
df[['dialogue','cleaned_dialogue']].sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dialogue</th>
      <th>cleaned_dialogue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14684</th>
      <td>Thank you.</td>
      <td>thank</td>
    </tr>
    <tr>
      <th>58592</th>
      <td>Hi tough guy.  I guess it worked huh?</td>
      <td>hi tough guy guess worked huh</td>
    </tr>
    <tr>
      <th>209279</th>
      <td>I've decided not to open a practice here  I wa...</td>
      <td>decided open practice want set research clinic...</td>
    </tr>
    <tr>
      <th>95420</th>
      <td>Am I suppose to be this sore?</td>
      <td>suppose sore</td>
    </tr>
    <tr>
      <th>50378</th>
      <td>You could still always give Becker an itch. 'C...</td>
      <td>could still always give becker itch course mig...</td>
    </tr>
  </tbody>
</table>
</div>



### Create training dataset
Now, we can aggregate all data for a particular movie character into 1 record. We will combine their dialogue tokens from the entire movie, calculate their median dialogue length by characters & words, and count their total no of lines in the movie.


```python
train = df.groupby(['chId', 'mId', 'chName', 'gender', 'posCredits','releaseYear']). \
            agg({'lineLength' : ['median'], 
                 'wordCountLine' : ['median'],
                 'chId' : ['count'],
                 'cleaned_dialogue' : [lambda x : ' '.join(x)]
                })

## Renaming columns by aggregate functions
train.columns = ["_".join(x) for x in train.columns.ravel()]

train.reset_index(inplace=True)
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chId</th>
      <th>mId</th>
      <th>chName</th>
      <th>gender</th>
      <th>posCredits</th>
      <th>releaseYear</th>
      <th>lineLength_median</th>
      <th>wordCountLine_median</th>
      <th>chId_count</th>
      <th>cleaned_dialogue_&lt;lambda&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>u0</td>
      <td>m0</td>
      <td>BIANCA</td>
      <td>1</td>
      <td>4</td>
      <td>1999.0</td>
      <td>34.0</td>
      <td>7.0</td>
      <td>94</td>
      <td>hope let go okay gonna need learn lie like fe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>u100</td>
      <td>m6</td>
      <td>AMY</td>
      <td>1</td>
      <td>7</td>
      <td>1999.0</td>
      <td>23.0</td>
      <td>4.0</td>
      <td>31</td>
      <td>died sleep three day ago paper tom dead  calli...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>u1003</td>
      <td>m65</td>
      <td>RICHARD</td>
      <td>0</td>
      <td>3</td>
      <td>1996.0</td>
      <td>24.5</td>
      <td>5.0</td>
      <td>70</td>
      <td>asked would said room room serious foolin arou...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>u1005</td>
      <td>m65</td>
      <td>SETH</td>
      <td>0</td>
      <td>2</td>
      <td>1996.0</td>
      <td>37.0</td>
      <td>8.0</td>
      <td>163</td>
      <td>let follow said new jesus christ carlos brothe...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>u1008</td>
      <td>m66</td>
      <td>C.O.</td>
      <td>0</td>
      <td>10+</td>
      <td>1997.0</td>
      <td>48.0</td>
      <td>9.0</td>
      <td>33</td>
      <td>course uh v p security arrangement generally t...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2946</th>
      <td>u980</td>
      <td>m63</td>
      <td>VICTOR</td>
      <td>0</td>
      <td>3</td>
      <td>1931.0</td>
      <td>32.0</td>
      <td>6.0</td>
      <td>126</td>
      <td>never said name remembers  kill draw line take...</td>
    </tr>
    <tr>
      <th>2947</th>
      <td>u983</td>
      <td>m64</td>
      <td>ALICE</td>
      <td>1</td>
      <td>10+</td>
      <td>2009.0</td>
      <td>30.0</td>
      <td>6.0</td>
      <td>51</td>
      <td>maybe wait mr christy killer still bill bill b...</td>
    </tr>
    <tr>
      <th>2948</th>
      <td>u985</td>
      <td>m64</td>
      <td>BILL</td>
      <td>0</td>
      <td>10+</td>
      <td>2009.0</td>
      <td>20.0</td>
      <td>4.0</td>
      <td>39</td>
      <td>twenty mile crossroad steve back hour thing st...</td>
    </tr>
    <tr>
      <th>2949</th>
      <td>u997</td>
      <td>m65</td>
      <td>JACOB</td>
      <td>0</td>
      <td>1</td>
      <td>1996.0</td>
      <td>36.0</td>
      <td>6.0</td>
      <td>90</td>
      <td>meant son daughter oh daughter bathroom vacati...</td>
    </tr>
    <tr>
      <th>2950</th>
      <td>u998</td>
      <td>m65</td>
      <td>KATE</td>
      <td>1</td>
      <td>4</td>
      <td>1996.0</td>
      <td>20.0</td>
      <td>4.5</td>
      <td>44</td>
      <td>everybody go home going em swear god father ch...</td>
    </tr>
  </tbody>
</table>
<p>2951 rows × 10 columns</p>
</div>



### Let's check some feature distributions by gender


```python
sns.boxplot(data = train, x = 'gender', y = 'chId_count', hue = 'gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe92d3d57d0>




![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_35_1.png)


The `chId_count` here refers to the no of lines given to the character in the movie. While the median value seems to be roughly similar for both males & females, the upper bound seems to be higher for males.

```python
sns.boxplot(data = train, x = 'gender', y = 'wordCountLine_median', hue = 'gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe92d108550>




![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_36_1.png)


The count of words per dialogue is higher for male characters than that for female characters!

```python
sns.boxplot(data = train, x = 'gender', y = 'lineLength_median', hue = 'gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe92d0a1dd0>




![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_37_1.png)


The median length of a dialogue also seems to be higher for males. 

```python
sns.scatterplot(data = train, x = 'wordCountLine_median', y = 'chId_count', hue = 'gender', alpha = 0.5) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe93902b810>




![png](/assets/images//gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_38_1.png)

Again, in the scatter plot, we see female characters, ie yellow points, generally closer to the origin, as they have smaller dialogues & lesser dialogues per movie, while male characters denoted by blue dots are more outward from the origin.

### Train test split

Now, we can split our data into a training set & a validation set.


```python
## Separating labels from features
y = train['gender']
X = train.copy()
X.drop('gender', axis=1, inplace=True)

## Removing unnecessary columns
X.drop('chId', axis=1, inplace=True)
X.drop('mId', axis=1, inplace=True)
X.drop('chName', axis=1, inplace=True)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>posCredits</th>
      <th>releaseYear</th>
      <th>lineLength_median</th>
      <th>wordCountLine_median</th>
      <th>chId_count</th>
      <th>cleaned_dialogue_&lt;lambda&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>1999.0</td>
      <td>34.0</td>
      <td>7.0</td>
      <td>94</td>
      <td>hope let go okay gonna need learn lie like fe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1999.0</td>
      <td>23.0</td>
      <td>4.0</td>
      <td>31</td>
      <td>died sleep three day ago paper tom dead  calli...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1996.0</td>
      <td>24.5</td>
      <td>5.0</td>
      <td>70</td>
      <td>asked would said room room serious foolin arou...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1996.0</td>
      <td>37.0</td>
      <td>8.0</td>
      <td>163</td>
      <td>let follow said new jesus christ carlos brothe...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10+</td>
      <td>1997.0</td>
      <td>48.0</td>
      <td>9.0</td>
      <td>33</td>
      <td>course uh v p security arrangement generally t...</td>
    </tr>
  </tbody>
</table>
</div>



We will pick equal no of records for both male & female characters to avoid any kind of bias due to no of records.


```python
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X, y)
y_under.value_counts()
```




    1    948
    0    948
    Name: gender, dtype: int64



We'll also try to keep equal no of male & female records in the train & validation datasets


```python
X_train, X_val, y_train, y_val = train_test_split(X_under, y_under, test_size=0.2, random_state = 10, stratify=y_under)

y_val.value_counts()
```




    1    190
    0    190
    Name: gender, dtype: int64




```python
X_val.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>posCredits</th>
      <th>releaseYear</th>
      <th>lineLength_median</th>
      <th>wordCountLine_median</th>
      <th>chId_count</th>
      <th>cleaned_dialogue_&lt;lambda&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1236</th>
      <td>2</td>
      <td>2001.0</td>
      <td>33.0</td>
      <td>6.0</td>
      <td>60</td>
      <td>latitude degree  maybe satellite said three ye...</td>
    </tr>
    <tr>
      <th>924</th>
      <td>10+</td>
      <td>1974.0</td>
      <td>23.0</td>
      <td>5.0</td>
      <td>23</td>
      <td>sure okay okay got boat plus owe know oh gee m...</td>
    </tr>
    <tr>
      <th>868</th>
      <td>4</td>
      <td>2001.0</td>
      <td>34.0</td>
      <td>6.0</td>
      <td>34</td>
      <td>going coast alan idea alive headed need stick...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>1</td>
      <td>1999.0</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>146</td>
      <td>poor woman carole wound could hope pacify evas...</td>
    </tr>
    <tr>
      <th>989</th>
      <td>10+</td>
      <td>2000.0</td>
      <td>42.0</td>
      <td>8.0</td>
      <td>15</td>
      <td>okay took trouble come got principle selling o...</td>
    </tr>
  </tbody>
</table>
</div>



### Pipeline for classifiers

Since our dataset includes both numerical features & NLP tokens, we'll use a special converter class in our pipeline.


```python
class Converter(BaseEstimator, TransformerMixin):
    ## Source : https://www.kaggle.com/tylersullivan/classifying-phishing-urls-three-models
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()
```

Pipeline for numeric features


```python
numeric_features = ['lineLength_median', 'wordCountLine_median', 'chId_count', 'releaseYear']

numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
```

Pipeline for tokens dereived from dialogues


```python
categorical_features = ['posCredits']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```


```python
vectorizer_features = ['cleaned_dialogue_<lambda>']
vectorizer_transformer = Pipeline(steps=[
    ('con', Converter()),
    ('tf', TfidfVectorizer())])
```

Now, we can combine preprocessing pipelines with the classifers. We will try 4 basic models:
- Linear Support Vector Classifier
- Logistic Regression Classifier
- Naive Bayes Classifier
- Random Forest Clasifier


```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('vec', vectorizer_transformer, vectorizer_features)
    ])

svc_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', CalibratedClassifierCV(LinearSVC()))])  ## LinearSVC has no predict_proba method

log_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

nb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', MultinomialNB())])

rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=120, min_samples_leaf=10, 
                                                            max_features=0.7, n_jobs=-1, oob_score=True))])
```

Fitting the preprocessing & classifier pipelines on training data


```python
svc_clf.fit(X_train, y_train)
log_clf.fit(X_train, y_train)
nb_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
```




    Pipeline(steps=[('preprocessor',
                     ColumnTransformer(transformers=[('num',
                                                      Pipeline(steps=[('scaler',
                                                                       MinMaxScaler())]),
                                                      ['lineLength_median',
                                                       'wordCountLine_median',
                                                       'chId_count',
                                                       'releaseYear']),
                                                     ('cat',
                                                      Pipeline(steps=[('onehot',
                                                                       OneHotEncoder(handle_unknown='ignore'))]),
                                                      ['posCredits']),
                                                     ('vec',
                                                      Pipeline(steps=[('con',
                                                                       Converter()),
                                                                      ('tf',
                                                                       TfidfVectorizer())]),
                                                      ['cleaned_dialogue_<lambda>'])])),
                    ('classifier',
                     RandomForestClassifier(max_features=0.7, min_samples_leaf=10,
                                            n_estimators=120, n_jobs=-1,
                                            oob_score=True))])



### Check results on the validation set


```python
def results(name: str, model: BaseEstimator) -> None:
    '''
    Custom function to check model performance on validation set
    '''
    preds = model.predict(X_val)

    print(name + " score: %.3f" % model.score(X_val, y_val))
    print(classification_report(y_val, preds))
    labels = ['Male', 'Female']

    conf_matrix = confusion_matrix(y_val, preds)
    plt.figure(figsize= (10,6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix for " + name)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
```


```python
results("SVC" , svc_clf)
results("Logistic Regression" , log_clf)
results("Naive Bayes" , nb_clf)
results("Random Forest" , rf_clf)
```

    SVC score: 0.787
                  precision    recall  f1-score   support
    
               0       0.77      0.82      0.79       190
               1       0.80      0.76      0.78       190
    
        accuracy                           0.79       380
       macro avg       0.79      0.79      0.79       380
    weighted avg       0.79      0.79      0.79       380
    
    Logistic Regression score: 0.768
                  precision    recall  f1-score   support
    
               0       0.77      0.77      0.77       190
               1       0.77      0.76      0.77       190
    
        accuracy                           0.77       380
       macro avg       0.77      0.77      0.77       380
    weighted avg       0.77      0.77      0.77       380
    
    Naive Bayes score: 0.761
                  precision    recall  f1-score   support
    
               0       0.74      0.80      0.77       190
               1       0.78      0.72      0.75       190
    
        accuracy                           0.76       380
       macro avg       0.76      0.76      0.76       380
    weighted avg       0.76      0.76      0.76       380
    
    Random Forest score: 0.721
                  precision    recall  f1-score   support
    
               0       0.72      0.72      0.72       190
               1       0.72      0.72      0.72       190
    
        accuracy                           0.72       380
       macro avg       0.72      0.72      0.72       380
    weighted avg       0.72      0.72      0.72       380
    






![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_59_1.png)



![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_59_2.png)



![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_59_3.png)



![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_59_4.png)


We see that Linear SVC performs the best classification with an accuracy & F1 score of ~79% !!

From the confusion matrix, we can see that out of the 190 male characters in the validation dataset, SVC model classified 155 of them correctly as males, and the remaining 35 incorrectly as females. Similarly, out of 190 female characters in the validation dataset, 144 were classified correctly & 46 classified incorrectly. 

Logistic Regression & Naive Bayes classifiers are close at 77% & 76% accuracies respectively.
These results are not close to state of the art, but are still pretty good.

Let's now explore what features contribute the most to our classifiers performance through some model explainability techniques.

## Feature importance

Creating a list of all features including numeric, categorical & vectorised features.


```python
vect_columns = list(svc_clf.named_steps['preprocessor'].named_transformers_['vec'].named_steps['tf'].get_feature_names())
onehot_columns = list(svc_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(input_features=categorical_features))
numeric_features_list = list(numeric_features)
numeric_features_list.extend(onehot_columns)
numeric_features_list.extend(vect_columns)
```

#### Feature importance for Logistic Regression


```python
lr_weights = eli5.explain_weights_df(log_clf.named_steps['classifier'], top=30, feature_names=numeric_features_list)
lr_weights.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>feature</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>oh</td>
      <td>2.914220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>really</td>
      <td>1.598428</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>love</td>
      <td>1.582584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>hi</td>
      <td>1.173297</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>said</td>
      <td>1.161967</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>want</td>
      <td>1.053116</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>like</td>
      <td>1.020965</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>darling</td>
      <td>0.992410</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>never</td>
      <td>0.987203</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>child</td>
      <td>0.975300</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>please</td>
      <td>0.970647</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>god</td>
      <td>0.941234</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>know</td>
      <td>0.913415</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>honey</td>
      <td>0.903731</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>school</td>
      <td>0.897518</td>
    </tr>
  </tbody>
</table>
</div>




```python
lr_weights.tail(14)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>feature</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>son</td>
      <td>-0.876330</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>good</td>
      <td>-0.897884</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>right</td>
      <td>-0.899643</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>chId_count</td>
      <td>-0.916359</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>fuck</td>
      <td>-0.996575</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>fuckin</td>
      <td>-1.049543</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>yeah</td>
      <td>-1.091158</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>hell</td>
      <td>-1.137874</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>got</td>
      <td>-1.162604</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>shit</td>
      <td>-1.162634</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>sir</td>
      <td>-1.201654</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>gotta</td>
      <td>-1.246364</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>hey</td>
      <td>-1.549047</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>man</td>
      <td>-2.188670</td>
    </tr>
  </tbody>
</table>
</div>



We see that dialogue keywords like `oh`, `love`, `like`, `darling`, `want`, `honey` are strong indicators that the character is a female, while keywords like `son`, `sir`, `man`, `hell`, `gotta`, `yeah` & most cuss words are usually found in the dialogues of male characters of the given Hollywood movies!

#### Let's also try to visualize a single decision tree

We can training a single decision tree using the Random Forest Classifier.


```python
m = RandomForestClassifier(n_estimators=1, min_samples_leaf=5, max_depth = 3, 
                           oob_score=True, random_state = np.random.seed(123))
dt_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', m)])

dt_clf.fit(X_train, y_train)
results("Decision Tree Classifier", dt_clf)
```

    Decision Tree Classifier score: 0.526
                  precision    recall  f1-score   support
    
               0       0.54      0.37      0.44       190
               1       0.52      0.68      0.59       190
    
        accuracy                           0.53       380
       macro avg       0.53      0.53      0.51       380
    weighted avg       0.53      0.53      0.51       380
    



![png](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_69_1.png)


While a single decision is a poor classifier with accuracy barely more than 50%, we see that bagging enough of such weak classifiers to form a Random Forest model helps us improve the model performance drastically! Let's look at how the splits are made for a single decision tree.


```python
'''
def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ 
    Draws a representation of a decition tree in IPython. Source : fastai v0.7
    Have commented the function definition here due to a Jekyll build error related to Liquid objects.
    """
    s=export_graphviz(t, out_file=None, feature_names=numeric_features_list, filled=True,
                      special_characters=True, rotate=True, precision=precision, 
                      proportion=True, class_names = ["male", "female"], impurity = False)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree size={size}; ratio={ratio}', s)))
'''

draw_tree(m.estimators_[0], X_train, precision=2)
```


![svg](/assets/images/gender-classifier-based-on-movie-dialogues_files/gender-classifier-based-on-movie-dialogues_71_0.svg)


Here, the blue coloured nodes indicate their majority class is `female` while the orange colored nodes have a majority of `male` labels. The decision tree starts with a mixed sample, but the leaves of the tree are biased towards one class or the other. Most splits seem to be happening using dialogue tokens. For eg., in the above tree, if the tf-idf frequency of keywords `think` is > 0.1 & `kid` is > 0.03, the samples are classified as `female`. 

#### Feature importance for the Random Forest model


```python
eli5.explain_weights_df(rf_clf.named_steps['classifier'], top=30, feature_names=numeric_features_list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>weight</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>oh</td>
      <td>0.077328</td>
      <td>0.027570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>man</td>
      <td>0.040615</td>
      <td>0.030255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>love</td>
      <td>0.022664</td>
      <td>0.026011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>shit</td>
      <td>0.019557</td>
      <td>0.027031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>said</td>
      <td>0.017135</td>
      <td>0.018359</td>
    </tr>
    <tr>
      <th>5</th>
      <td>lineLength_median</td>
      <td>0.015757</td>
      <td>0.017705</td>
    </tr>
    <tr>
      <th>6</th>
      <td>got</td>
      <td>0.013712</td>
      <td>0.018381</td>
    </tr>
    <tr>
      <th>7</th>
      <td>really</td>
      <td>0.013227</td>
      <td>0.018366</td>
    </tr>
    <tr>
      <th>8</th>
      <td>hey</td>
      <td>0.012435</td>
      <td>0.019035</td>
    </tr>
    <tr>
      <th>9</th>
      <td>good</td>
      <td>0.012253</td>
      <td>0.017000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>look</td>
      <td>0.011174</td>
      <td>0.016425</td>
    </tr>
    <tr>
      <th>11</th>
      <td>right</td>
      <td>0.009775</td>
      <td>0.012608</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sir</td>
      <td>0.009673</td>
      <td>0.018090</td>
    </tr>
    <tr>
      <th>13</th>
      <td>think</td>
      <td>0.009661</td>
      <td>0.014078</td>
    </tr>
    <tr>
      <th>14</th>
      <td>know</td>
      <td>0.009660</td>
      <td>0.012629</td>
    </tr>
    <tr>
      <th>15</th>
      <td>em</td>
      <td>0.008794</td>
      <td>0.019125</td>
    </tr>
    <tr>
      <th>16</th>
      <td>like</td>
      <td>0.008730</td>
      <td>0.011206</td>
    </tr>
    <tr>
      <th>17</th>
      <td>understand</td>
      <td>0.008263</td>
      <td>0.013995</td>
    </tr>
    <tr>
      <th>18</th>
      <td>want</td>
      <td>0.008112</td>
      <td>0.012310</td>
    </tr>
    <tr>
      <th>19</th>
      <td>yeah</td>
      <td>0.007471</td>
      <td>0.014095</td>
    </tr>
    <tr>
      <th>20</th>
      <td>get</td>
      <td>0.007464</td>
      <td>0.010990</td>
    </tr>
    <tr>
      <th>21</th>
      <td>would</td>
      <td>0.007399</td>
      <td>0.010665</td>
    </tr>
    <tr>
      <th>22</th>
      <td>chId_count</td>
      <td>0.007074</td>
      <td>0.010570</td>
    </tr>
    <tr>
      <th>23</th>
      <td>come</td>
      <td>0.006845</td>
      <td>0.010507</td>
    </tr>
    <tr>
      <th>24</th>
      <td>god</td>
      <td>0.006782</td>
      <td>0.011838</td>
    </tr>
    <tr>
      <th>25</th>
      <td>releaseYear</td>
      <td>0.006504</td>
      <td>0.009230</td>
    </tr>
    <tr>
      <th>26</th>
      <td>hi</td>
      <td>0.006305</td>
      <td>0.016955</td>
    </tr>
    <tr>
      <th>27</th>
      <td>one</td>
      <td>0.006265</td>
      <td>0.009646</td>
    </tr>
    <tr>
      <th>28</th>
      <td>gotta</td>
      <td>0.006075</td>
      <td>0.014320</td>
    </tr>
    <tr>
      <th>29</th>
      <td>child</td>
      <td>0.005858</td>
      <td>0.013265</td>
    </tr>
  </tbody>
</table>
</div>



We see that the median length of a dialogue, total no of lines (`chId_count`) & movie release year are important features along with the tokens extracted from the character's dialogues for the Random Forest model!

### Next Steps

Some possible ways to further improve the classifier performance could be:
- using bi-grams or tri-grams for dialogue tokens
- Adding features related to sentiments extracted from dialogues
- Adding a feature that measures the level of objectivity or subjectivity of a dialogue
- hyper-parameter tuning of our model parameters
- trying out XGBoost or neural network models

Still, our current best model (Linear SVC) can classify roughly 4 out of 5 movie characters (79% accuracy) correctly using the dialogues they speak, and some movie metadata like release year and position of character in the movie credits. We can safely say that our model is able to capture the gender specific bias in the characters of Hollywood movies. 

If you would like to play around with the code, the complete Jupyter notebook is available [here](https://www.kaggle.com/priteshshrivastava/gender-classifier-based-on-movie-dialogues) on Kaggle.
