
# Networks analysis and prediction on ISIS twitter network: Part 2

Dataset from: https://www.kaggle.com/fifthtribe/how-isis-uses-twitter
This project utilizes the above dataset of pro-ISIS tweets collected in a period during 2014 to 2015. 

---

## Part Two : Models

(Continued from Part 1)   

I originally wanted to predict ISIS attacks with this data, but this was deemed unfeasible as the tweets' content were mainly centered on commentary on Middle East news, as well as sharing of propaganda. However, I thought it would be interesting to predict the intent of tweets based on the following categories: 

- N: News. Informative in nature with statements on curent events or links to news.
- P: Propaganda. Links to ISIS news sources, or biased links.
- O: Opinion. Containing personal thoughts. Religious quotes were placed here as well as they constitute religious opinion.

Previously, duplicates were removed by detecting for 5-grams in tweets content. Unique tweets were then sampled from the dataset and manually labelled. 


```python
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import math
import datetime
import pickle

import re
# import textblob
from langdetect import detect

from collections import Counter

%matplotlib inline
style.use('fivethirtyeight')
```


```python
labelled_csv = 'uniquetweets.csv'
labelled_tweets = pd.read_csv(labelled_csv)
```


```python
labelled_tweets.head()
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
      <th>name</th>
      <th>username</th>
      <th>description</th>
      <th>location</th>
      <th>followers</th>
      <th>numberstatuses</th>
      <th>time</th>
      <th>tweets</th>
      <th>Label</th>
      <th>url</th>
      <th>lang</th>
      <th>mentions</th>
      <th>hashtags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Al-Battar English</td>
      <td>Al_Battar_Engl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>117</td>
      <td>144</td>
      <td>30/1/2016 20:00</td>
      <td>#IslamicState\r\n#WilayatAlKhyar\r\nImages of ...</td>
      <td>P</td>
      <td>['https://t.co/sVS9Vxwjtt']</td>
      <td>en</td>
      <td>[]</td>
      <td>['#IslamicState', '#WilayatAlKhyar', '#Martydo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Asim Abu Merjem</td>
      <td>AsimAbuMerjem</td>
      <td>Servant of Allah, in need of Allah's mercy !!!</td>
      <td>NaN</td>
      <td>742</td>
      <td>348</td>
      <td>17/1/2016 22:00</td>
      <td>Massive TACTICAL RETREAT of "brave" SAA soldi...</td>
      <td>O</td>
      <td>[]</td>
      <td>en</td>
      <td>['@leithfadel']</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marwan Qassami</td>
      <td>QassamiMarwan</td>
      <td>Humanitarian, social entrepreneur, Independent...</td>
      <td>Antas, Bahia</td>
      <td>1593</td>
      <td>908</td>
      <td>29/1/2016 18:00</td>
      <td>Western secular humanist values:\r\n</td>
      <td>O</td>
      <td>['https://t.co/IMfvB8ZDqd']</td>
      <td>en</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>General</td>
      <td>ismailmahsud</td>
      <td>Listen! No affiliations, Final year research o...</td>
      <td>S.Wazirstan|Mahsud not a Wazir</td>
      <td>392</td>
      <td>743</td>
      <td>28/1/2016 8:00</td>
      <td>#سامراء tough clashes still ongoing around. #I...</td>
      <td>N</td>
      <td>[]</td>
      <td>en</td>
      <td>[]</td>
      <td>['#سامراء', '#Iraq']</td>
    </tr>
    <tr>
      <th>6</th>
      <td>War Reporter</td>
      <td>warreporter2</td>
      <td>Reporting, analysing and discussing conflicts ...</td>
      <td>München, Deutschland</td>
      <td>139</td>
      <td>656</td>
      <td>14/1/2016 12:00</td>
      <td>RT : #Syria path of Tawheed - from the officia...</td>
      <td>P</td>
      <td>[]</td>
      <td>en</td>
      <td>['@thevictoryseri4']</td>
      <td>['#Syria']</td>
    </tr>
  </tbody>
</table>
</div>



As the dataset was manually sampled, it is now relatively small compared to the original. 


```python
len(labelled_tweets)
```




    255



The dataset looks pretty balanced, not much difference in count between the different Labels.


```python
sns.countplot(labelled_tweets['Label'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d1f43ca90>




![png](/assets/ISIS-Project-2/output_7_1.png){:class="img-responsive"}


Baseline accuracy as calculated from the largest class. 


```python
labelled_tweets['Label'].value_counts()['O'] / float(len(labelled_tweets))
```




    0.36862745098039218



---
### Construct a dictionary of top hashtags to vectorize the tweets

Here, the hashtags were vectorised based on their counts.Chi-squared was then used to select the top 3 features.   
A good intuition and example for chi-squared can be found here: <https://chrisalbon.com/machine_learning/feature_selection/chi-squared_for_feature_selection/>


```python
from sklearn.feature_extraction.text import CountVectorizer
```

The function below retrieves strings from a nested list, and removes empty lists. 


```python
def flatten_string(user_list):
    return[user.replace(' ','') for y in user_list if y for user in y]
```


```python
hashtags = [hashtag for hashtag in labelled_tweets['hashtags'] if hashtag]
all_hashtags = flatten_string(hashtags)
hashcount = pd.Series(Counter(all_hashtags))
```


```python
countvec = CountVectorizer(lowercase=True)
cv_matrix = countvec.fit_transform(Xtrain['hashtags'])
```

'hash_cols' only gives us the column number of the feature in the matrix returned by CountVectorizer. The column names are then retrieved by using 'countvec.get_feature_names()'.


```python
# column numbers in cv_matrix - top 3 hashtags for predicting Labels based on chi2
hash_cols = select.get_support().nonzero()
for i in hash_cols[0]: 
    print (countvec.get_feature_names()[i])
```

    anbar
    assad
    isis
    

The relevant subset of the word count matrix was then retrieved for words selected through chi-squared.


```python
cv_subset = cv_matrix[:,hash_cols[0]]
```


```python
besthash = [countvec.get_feature_names()[i] for i in hash_cols[0]]

tophashdf = pd.DataFrame(cv_subset.A, 
                         columns=besthash,
                         index=Xtrain.index)
tophashdf.head()
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
      <th>anbar</th>
      <th>assad</th>
      <th>isis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



A dataframe of the respective word counts from the hashtags was then constructed. This was repeated for the test set. 

####  Extracting the word count from tweets, and not just hashtags

In addition to selecting features from the hashtags used, unigrams were also selected from the tweets themselves. Again, a chi-squared was used to select the top 10 features. 


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

ytraindf = pd.DataFrame({'Labels': ytrain})

ytraindf['News'] = [1 if tweet=='N' else 0 for tweet in ytrain]
ytraindf['Opinion'] = [1 if tweet=='O' else 0 for tweet in ytrain]

select = SelectKBest(chi2, k=10)
select.fit(cv_matrix, ytraindf.drop(['Labels'], axis=1))
```




    SelectKBest(k=10, score_func=<function chi2 at 0x0000013D1F3A5D08>)




```python
# column numbers in cv_matrix - top 3 hashtags for predicting Labels based on chi2
ugram_cols = select.get_support().nonzero()
ugram_cols_names = []
for i in ugram_cols[0]: 
    print (countvec.get_feature_names()[i])
    ugram_cols_names.append(countvec.get_feature_names()[i])
```

    allah
    anbar
    by
    desire
    in
    killed
    like
    me
    we
    you
    

The words selected from tweets were more expressive. For example, words like 'desire', 'me', 'we' and 'you' are more expressive of personal sentiment whereas words like 'killed' and 'by' might appear more often in the news and propaganda class. Again, this was also replicated on the test set.


```python
cv_subset = cv_matrix[:,ugram_cols[0]]
ugramdf = pd.DataFrame(cv_subset.A, 
                     columns=ugram_cols_names,
                     index=Xtrain.index)
```


```python
ugramdf.head()
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
      <th>allah</th>
      <th>anbar</th>
      <th>by</th>
      <th>desire</th>
      <th>in</th>
      <th>killed</th>
      <th>like</th>
      <th>me</th>
      <th>we</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Adding more features related to ISIS propaganda

Amaq News Agency is a news outlet linked with ISIS. '#amaqnewsagency', '#amaq' or '#amaqnews' would then be relevant hashtags in indicating if there is a tweet with intention of spreading propaganda. This is known and definitely generalizable - it should be included as a feature regardless of what chi2 says.


```python
hashlist = ['#amaq', '#caliphate_news']
def custom_hash(hashlist, X):
    return [1 if any(hashtag in hashtags.lower() for hashtag in hashlist) else 0 for hashtags in X]
```


```python
Xtrain['amaq_mention'] = custom_hash(hashlist, Xtrain['hashtags'])
Xtest['amaq_mention'] = custom_hash(hashlist, Xtest['hashtags'])
```

Additionally, we also know that territorial claims made by ISIS are called 'wilayat'. A search on jihadology.net, a website by Aaron Zelin, a Washington researcher on jihadi groups, turns up many ISIS videos with the typical title starting with 'Wilayat-Al-' followed by an Arabic phrase. Hence, hashtags containing 'wilayat' in this context could be indicative of propaganda.


```python
hashlist2 = ['wilayat']

Xtrain['wilayat_mention'] = custom_hash(hashlist2, Xtrain['hashtags'])
Xtest['wilayat_mention'] = custom_hash(hashlist2, Xtest['hashtags'])
```

---
### POS Tagging 

Part-of-speech tagging was done on the tweets, using the spaCy library. POS tagging identifies the grammatic category of each word (its part of speech - for example, a noun or verb), and used here to tease out the difference in syntactic structure between opinions, news, and propaganda. 


```python
import spacy
nlp = spacy.load('en')
```


```python
# consolidated list of tags for each tweet
def all_tags(col):
    return [[word.pos_ for word in nlp(tweet)] for tweet in col]
```


```python
Xtrain['POStags'] = all_tags(Xtrain['tweets'])
Xtest['POStags'] = all_tags(Xtest['tweets'])
```


```python
# function adds number of tags in tweet to a list
from collections import Counter

def add_tag(tag,df_col):    
    return [Counter(x)[tag] for x in df_col]
```


```python
Xtrain['propn_count'] = add_tag('PROPN', Xtrain['POStags'])
```

In addition to proper nouns, the above was repeated for other part of speech tags: nouns, verbs, adjectives, adverbs and pronouns. 

Next, use a chi2 squared test to choose 3 best features. My guess is that pronouns (I, you, he, she, themselves) are closely associated with opinions. I also noticed that the sentence structures between propaganda seem to be similar to that of news - attacks events by ISIS are reported matter-of-factly in a tone similar to news headlines. Hence, this might not be as useful for distinguishing between propaganda and news. Let's see what chi2 returns us.


```python
besttags = select.get_support().nonzero()[0]

Xpos.iloc[:,besttags].head()
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
      <th>propn_count</th>
      <th>adv_count</th>
      <th>pron_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>195</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



---
With all the features above ready, we begin the modelling.   
Each model below has two versions - the first one only using data from hashtags and parts of speech, while the second one included data from the content of tweets.

### First Model - Naives Bayes Classifier

Version 1: Using data from hashtags + POS


```python
from sklearn.naive_bayes import MultinomialNB

Xtrain_features = pd.concat([Xpos,
                             tophashdf,
                             Xtrain.loc[:,['amaq_mention','wilayat_mention']]],axis=1)
Xtest_features = pd.concat([Xpos_test,
                            tophashdf_test,
                            Xtest.loc[:,['amaq_mention','wilayat_mention']]],axis=1)


mnb = MultinomialNB()
model = mnb.fit(Xtrain_features, ytrain)
```


```python
y_pred = mnb.predict(Xtest_features)
```


```python
from sklearn.metrics import classification_report

print(classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.54      0.30      0.39        23
              O       0.73      0.79      0.76        28
              P       0.53      0.69      0.60        26
    
    avg / total       0.61      0.61      0.59        77
    
    


```python
# gridsearched with range of C values (best one below) However, the results did not improve by much. 
# results were also marginally better when more features were included.

model.best_params_
```




    {'alpha': 0.1}



A support vector classifier was built too. Results for opinion was better (70% precision and 80% recall) but not for news and propaganda. This could mean that it is harder to find a clear decision boundary between these 2 classes.

Version 2: Including data from tweets.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

Xtrain_features = pd.concat([ugramdf,
                             Xtrain.loc[:,['amaq_mention','wilayat_mention','propn_count', 'noun_count','verb_count','adj_count','adv_count','pron_count']]],axis=1)
Xtest_features = pd.concat([ugramdf_test,
                            Xtest.loc[:,['amaq_mention','wilayat_mention','propn_count', 'noun_count','verb_count','adj_count','adv_count','pron_count']]],axis=1)


mnb = MultinomialNB()
model = mnb.fit(Xtrain_features, ytrain)
```


```python
y_pred = mnb.predict(Xtest_features)

print(classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.67      0.35      0.46        23
              O       0.76      0.79      0.77        28
              P       0.56      0.77      0.65        26
    
    avg / total       0.66      0.65      0.64        77
    
    

### Second Model: Logistic Regression

Version 1: Using data from hashtags + POS


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=2018)

C = [1,5,10]
mc = ['multinomial','ovr']
solver = ['newton-cg','lbfgs']
param_grid = dict(class_weight=[None], C=C, multi_class=mc, solver=solver)

grid = GridSearchCV(lr, param_grid, cv=3)
```


```python
model = grid.fit(Xtrain_features, ytrain)
```


```python
y_pred = model.predict(Xtest_features)
```


```python
print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.44      0.17      0.25        23
              O       0.67      0.86      0.75        28
              P       0.53      0.65      0.59        26
    
    avg / total       0.55      0.58      0.55        77
    
    


```python
model.best_params_
```




    {'C': 1, 'class_weight': None, 'multi_class': 'ovr', 'solver': 'newton-cg'}



Version 2: Including data from tweets. 


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr = LogisticRegression(random_state=2018)

C = [1,5,10]
mc = ['multinomial','ovr']
solver = ['newton-cg','lbfgs']
param_grid = dict(class_weight=[None], C=C, multi_class=mc, solver=solver)

grid = GridSearchCV(lr, param_grid, cv=3)
model = grid.fit(Xtrain_features, ytrain)

y_pred = model.predict(Xtest_features)

print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.67      0.26      0.38        23
              O       0.65      0.93      0.76        28
              P       0.61      0.65      0.63        26
    
    avg / total       0.64      0.64      0.60        77
    
    


```python
model.best_params_
```




    {'C': 1, 'class_weight': None, 'multi_class': 'ovr', 'solver': 'newton-cg'}



### Third model: Random Forest

An ensemble of decision trees.   
Version 1: Using data from hashtags + POS


```python
# Xpos.iloc[:,select.get_support().nonzero()[0]]
Xtrain_features = pd.concat([Xpos.iloc[:,select.get_support().nonzero()[0]],
                             tophashdf,
                             Xtrain.loc[:,['amaq_mention','wilayat_mention']]],axis=1)
Xtest_features = pd.concat([Xpos_test.iloc[:,select.get_support().nonzero()[0]],
                            tophashdf_test,
                            Xtest.loc[:,['amaq_mention','wilayat_mention']]],axis=1)
```


```python
from sklearn.ensemble import RandomForestClassifier

n_est = [3,5,7]
depth = [None, 3,5,7]
param_grid = dict(n_estimators=n_est, max_depth=depth)

rfc = RandomForestClassifier(random_state=2018)

grid = GridSearchCV(rfc, param_grid)
model = grid.fit(Xtrain_features, ytrain)
y_pred = model.predict(Xtest_features)
```


```python
model.best_params_
```




    {'max_depth': None, 'n_estimators': 3}




```python
print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.57      0.52      0.55        23
              O       0.74      0.71      0.73        28
              P       0.59      0.65      0.62        26
    
    avg / total       0.64      0.64      0.64        77
    
    

For Random Forest, scores were significantly better when number of features were reduced to those filtered after chi2. This makes sense as it the decision trees would be prone to overfit.

Version 2: Using data from tweets


```python
from sklearn.ensemble import RandomForestClassifier

n_est = [3,5,7,9]
depth = [None,3,5,7]
param_grid = dict(n_estimators=n_est, max_depth=depth)

rfc = RandomForestClassifier(random_state=2018)

grid = GridSearchCV(rfc, param_grid)
model = grid.fit(Xtrain_features, ytrain)
```


```python
y_pred = model.predict(Xtest_features)

print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.53      0.43      0.48        23
              O       0.75      0.86      0.80        28
              P       0.58      0.58      0.58        26
    
    avg / total       0.62      0.64      0.63        77
    
    


```python
model.best_params_
```




    {'max_depth': 3, 'n_estimators': 5}



### Voting Classifier (ensemble)

A bagging ensemble - logistic regression, naive bayes, and random forest.    
Version 1: Using data from hashtags + POS.


```python
Xtrain_features = pd.concat([Xpos,
                             tophashdf,
                             Xtrain.loc[:,['amaq_mention','wilayat_mention']]],axis=1)
Xtest_features = pd.concat([Xpos_test,
                            tophashdf_test,
                            Xtest.loc[:,['amaq_mention','wilayat_mention']]],axis=1)
```


```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

mnb = MultinomialNB()
svc = SVC(class_weight='balanced', C=5, kernel='linear', random_state=2018)
rfc = RandomForestClassifier(n_estimators=3, max_depth=None, random_state=2018)

vc = VotingClassifier(estimators = [('mnb',mnb), ('svc',svc), ('rfc',rfc)],
                     voting='hard',
                     weights=[1,1,2])
```


```python
model = vc.fit(Xtrain_features, ytrain)
y_pred = model.predict(Xtest_features)
```


```python
print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.41      0.39      0.40        23
              O       0.66      0.82      0.73        28
              P       0.60      0.46      0.52        26
    
    avg / total       0.56      0.57      0.56        77
    
    

For this voting classifier, results (for News and Propaganda in particular) did not improve. Generally, this means that models were bad at classifying the same instances. The variance in each model was not compensated for. 

Version 2: Using data from tweets.


```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

mnb = MultinomialNB()
svc = SVC(class_weight=None, C=1, kernel='linear', random_state=2018)
rfc = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=2018)
# ': 1, 'class_weight': None, 'multi_class': 'ovr', 'solver': 'newton-cg'}

vc = VotingClassifier(estimators = [('mnb',mnb), ('svc',svc), ('rfc',rfc)],
                     voting='hard',
                     weights=[1,1,2])

model = vc.fit(Xtrain_features, ytrain)
y_pred = model.predict(Xtest_features)

print (classification_report(ytest, y_pred))
```

                 precision    recall  f1-score   support
    
              N       0.57      0.57      0.57        23
              O       0.77      0.86      0.81        28
              P       0.61      0.54      0.57        26
    
    avg / total       0.66      0.66      0.66        77
    
    

---
### Evaluation of models

The plot type below tries to estimate the central tendency (or mean) of the variable within the 95% confidence interval, and the plots show the counts for pronouns, proper nouns, nouns, verbs and adjectives. 

When looking at the syntactic structure of tweets as shown below, it is hard to tell the difference between news and propaganda. 


```python
sns.pointplot(x='pron_count', y='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23737c39d68>



![png](/assets/ISIS-Project-2/output_80_1.png){:class="img-responsive"}



```python
ax = sns.pointplot(x='propn_count', y='Label', data=overall)
ax.set(xlabel='Proper noun counts', ylabel='Target')
plt.show()
```


![png](/assets/ISIS-Project-2/output_81_1.png){:class="img-responsive"}



```python
sns.pointplot(x='noun_count', y='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23737b3c128>



![png](/assets/ISIS-Project-2/output_82_1.png){:class="img-responsive"}




```python
sns.pointplot(x='adj_count', y='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23737ba7550>



![png](/assets/ISIS-Project-2/output_83_1.png){:class="img-responsive"}



One possible reason could be the similarity in factual tone between news and propaganda, an observation I made when manually labelling the dataset. This could explain the poorer precision and recall scores for these classes compared to opinion. In particular, a significant proportion of news gets misclassified as other classes. 

Another area of assessment is the usage of chi-squared as features selection. 

A brief explanation and usage of the chi-squared formula can be found here: <https://chrisalbon.com/machine_learning/feature_selection/chi-squared_for_feature_selection/>
Calculating chi-squared indicates the statistical significance of a hypothesis of independence, between the class and the target. In other words, we are also selecting features on the assumption that a the feature and class are dependent - the occurence of a feature makes the occurence of a class more likely. 

The graphs below show the variance in top features selected from hashtags. 


```python
sns.countplot(x='isis', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d1f678358>



![png](/assets/ISIS-Project-2/output_85_1.png){:class="img-responsive"}



```python
sns.countplot(x='assad', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d1f6780f0>



![png](/assets/ISIS-Project-2/output_86_1.png){:class="img-responsive"}


When we look at the plots above, one realises that that there is not much variance. This could be due to the inherent structure of the tweets themselves - most might not use hashtags much. This also explains why the models had marginally better performance when unigrams were extracted from the content of tweets, and not just hashtags. 


```python
sns.countplot(x='wilayat_mention', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d2070abe0>



![png](/assets/ISIS-Project-2/output_88_1.png){:class="img-responsive"}


Plotting graphs now based on features extracted from unigrams in tweets. 


```python
sns.countplot(x='killed', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d20e60550>



![png](/assets/ISIS-Project-2/output_90_1.png){:class="img-responsive"}



```python
sns.countplot(x='allah', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d20e7ed30>



![png](/assets/ISIS-Project-2/output_91_1.png){:class="img-responsive"}



```python
sns.countplot(x='by', hue='Label', data=overall)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13d20f04f28>



![png](/assets/ISIS-Project-2/output_92_1.png){:class="img-responsive"}


Additionally, because of the overall lack of variance in the features, SVM might not perform so well. There is no clear decision boundary, or 'distance' between points. This could also explain better performance on the Random Forest model, where random trees further split data into classes based on smaller local clusters.

### Further Improvements

A deep learning model can be explored. Because the length of each tweet is short, the input sequence will be able to capture any potential hidden features from the entire tweet. However, one cannot utilize pre-trained gloVe or word2vec vectors for the embedding layer - there are multiple arabic words translated to english (wilayat, amaq, khilafah) that might not be in the the pretrained vocabulary. As such, one would expect to use a trainable embedding layer. 
A potential difficulty in this case is the presence of URLs in tweets, which might have to be removed.

Something to also explore as a feature is the number of twitter followers, or the number of users followed. An interesting question here is if an influential twitter user would spread more propaganda. The relation between twitter popularity within the network, and the content of one's tweets can then be looked at. 
