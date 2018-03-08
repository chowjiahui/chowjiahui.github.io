
---

# Networks analysis and prediction on pro-ISIS twitter network: Part 1

Dataset from: <https://www.kaggle.com/fifthtribe/how-isis-uses-twitter>   
This project utilizes the above dataset of pro-ISIS tweets collected in a period during 2014 to 2015. 

---

## Part 1: Networks analysis

Exploratory data analysis was first done to understand behaviour of twitter users, followed by networks analysis to identify influential users. The visualizations are under the 'ISIS_tweets.ipynb' file, and are done with Gephi.    

I also made an interactive networks diagram of the pro-ISIS network, using Gephi's sigma-js plugin: 
<https://chowjiahui.github.io/ISISnetwork>    
More improvements can be done here as I have yet to reduce the borders of the nodes. 

Part 2 will focus on classification of the intent of tweets. 

Overall Steps: 
- Load and clean the data
- Retrieve potential features from the tweets - usernames, URLs, language used, hashtags
- Convert dates from string data type to datetime data type 
- Exploratory data analysis
- Networks analysis and visualization using Gephi

---

## Load the data.


Getting the encoding type of the csv file to open it correctly.


```python
import chardet

tweets_csv = './v1/tweets1.csv'

# tweets.to_csv(tweets_csv, encoding='utf-8')
```

Painful lesson: very important to open the csv in 'utf-8' encoding, and save it in that same encoding too. Renders the arabic text correctly. Reading the csv using the default engine gives a buffer overflow error.


```python
tweets = pd.read_csv(tweets_csv, encoding='utf-8', engine='python')
```


```python
tweets.head()
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
      <th>url</th>
      <th>lang</th>
      <th>mentions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GunsandCoffee</td>
      <td>GunsandCoffee70</td>
      <td>ENGLISH TRANSLATIONS: http://t.co/QLdJ0ftews</td>
      <td>NaN</td>
      <td>640.0</td>
      <td>49.0</td>
      <td>2015-01-06 21:00:07</td>
      <td>ENGLISH TRANSLATION: 'A MESSAGE TO THE TRUTHFU...</td>
      <td>['http://t.co/73xFszsjvr', 'http://t.co/x8BZcs...</td>
      <td>en</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GunsandCoffee</td>
      <td>GunsandCoffee70</td>
      <td>ENGLISH TRANSLATIONS: http://t.co/QLdJ0ftews</td>
      <td>NaN</td>
      <td>640.0</td>
      <td>49.0</td>
      <td>2015-01-06 21:00:27</td>
      <td>ENGLISH TRANSLATION: SHEIKH FATIH AL JAWLANI '...</td>
      <td>['http://t.co/uqqzXGgVTz', 'http://t.co/A7nbjw...</td>
      <td>en</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GunsandCoffee</td>
      <td>GunsandCoffee70</td>
      <td>ENGLISH TRANSLATIONS: http://t.co/QLdJ0ftews</td>
      <td>NaN</td>
      <td>640.0</td>
      <td>49.0</td>
      <td>2015-01-06 21:00:29</td>
      <td>ENGLISH TRANSLATION: FIRST AUDIO MEETING WITH ...</td>
      <td>['http://t.co/TgXT1GdGw7', 'http://t.co/ZuE8ei...</td>
      <td>de</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GunsandCoffee</td>
      <td>GunsandCoffee70</td>
      <td>ENGLISH TRANSLATIONS: http://t.co/QLdJ0ftews</td>
      <td>NaN</td>
      <td>640.0</td>
      <td>49.0</td>
      <td>2015-01-06 21:00:37</td>
      <td>ENGLISH TRANSLATION: SHEIKH NASIR AL WUHAYSHI ...</td>
      <td>['http://t.co/3qg5dKlIwr', 'http://t.co/7bqk1w...</td>
      <td>en</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GunsandCoffee</td>
      <td>GunsandCoffee70</td>
      <td>ENGLISH TRANSLATIONS: http://t.co/QLdJ0ftews</td>
      <td>NaN</td>
      <td>640.0</td>
      <td>49.0</td>
      <td>2015-01-06 21:00:45</td>
      <td>ENGLISH TRANSLATION: AQAP: 'RESPONSE TO SHEIKH...</td>
      <td>['http://t.co/2EYm9EymTe']</td>
      <td>en</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17260 entries, 0 to 17259
    Data columns (total 11 columns):
    name              17260 non-null object
    username          17260 non-null object
    description       14626 non-null object
    location          11356 non-null object
    followers         17260 non-null object
    numberstatuses    17260 non-null object
    time              17260 non-null object
    tweets            17260 non-null object
    url               17260 non-null object
    lang              17260 non-null object
    mentions          17260 non-null object
    dtypes: object(11)
    memory usage: 1.4+ MB
    

Dropping rows with empty tweet values and saving the csv file. 


```python
# tweets.drop(empty_tweets, inplace=True) d
tweets['tweets'].isnull().sum()
tweets.to_csv(tweets_csv, encoding='utf-8')
```

---
## Data Cleaning  and Preprocessing

Problems encountered (other than not knowing a word of arabic): 
- a preliminary check on the first 1000 tweets do show that langdetect hates URLs
- the langdetect algorithm  misclassified some english tweets as foreign languages when they contained URLs. 
- some tweets are in 2 languages (arabic and english, for example)

### Adding a 'Language' column

The code below takes a while to run, so after running once, the language column was appended to the tweets file. Catching cases where langdetect throws an error after failing to detect language.


```python
lg = []
err = []
for x in tweets['tweets']:
    try: 
        lg.append(detect(x))
    except: 
        lg.append('Language not detected') 
```


```python
tweets['lang'] = lg
tweets.to_csv(tweets_csv, encoding='utf-8'
```


```python
tweets['lang'].value_counts().head()
```




    en    14385
    ar      654
    fr      581
    id      432
    so      348
    Name: lang, dtype: int64



### Moving URLs in tweets to a separate column

Some tweets had URLs. These caused the langdetect algorithm to mistake it for some other language when in fact it is in English. So I've decided to move URLs to a separate column in case I ever drum up the courage to open those URLs. Ominous. This also means I need to re-run langdetect after the URLs were removed.


```python
# empty list if findall returns no match
tweets['url'] = [re.findall('(https://t.co/[\w]+|http://t.co/[\w]+)',str(x)) for x in tweets['tweets']]
tweets['tweets'] = [re.sub('(https://t.co/[\w]+|http://t.co/[\w]+)','',str(a)) for a in tweets['tweets']]
```

Running the langdetect algorithm again, and then saving all these changes to the original csv file. 

### Moving usernames in tweets to a separate column

This likely affects the langdetect algorithm as well, which means I have to run it again! Need to remove the '@' as well, to easily parse this into an edge list later on. 


```python
tweets['mentions'] = [re.findall('(@[\w]+)',str(x)) for x in tweets['tweets']]
tweets['tweets'] = [re.sub('(@[\w]+)','',str(a)) for a in tweets['tweets']]
```

There is a funny username that is absurdly long. Let's just use the display name as the username. 


```python
a = '````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````'
whereitis = tweets[tweets['username']==a].index
new_username= tweets[tweets['username']==a]['name'].iloc[0].replace(' ','')
tweets.loc[whereitis[0],['username']] = new_username
```

### Converting the date string to datetime

Extracting the month and day from the date column, which looks like a string for now

Converting date strings to datetime objects so that month and day values can easily be extracted later on. 


```python
dates = [datetime.strptime(a, '%Y-%m-%d %H:%M:%S') for a in tweets['time']]
tweets['time'] = dates
#tweets.to_csv(tweets_csv, encoding='utf-8')
```


```python
tweets.drop(tweets[tweets['time']=='en'].index, axis=0, inplace=True)
```


```python
tweets1['time'][0]
```




    Timestamp('2015-01-06 21:00:07')



### Extracting the hashtags 

This can be used as a feature later on for attack prediction.


```python
test = []
for tweet in tweets1['tweets']:
    try:
        test.append(re.findall('#\w+', tweet))
    except: 
        test.append(None)
```


```python
tweets1['hashtags'] = test
```

---
## EDA

### Subsequent plots explore these relationships: 
- tweets frequency by time, by most frequent users
- commonly mentioned words and topics 
- later on after network analysis is done, behaviour of key users further investigated


```python
tweets_by_day = tweets1.set_index('time').resample('D').size()
day = tweets_by_day.plot(figsize=(20,5))
day.set_ylabel('Number of tweets')
day.set_xlabel('2015 - 2016')
plt.show()
```

![png](/assets/ISIS-Project-1/output_30_0.jpg){:class='img-responsive'}


The tweeting activity increases drastically after the Paris bombing attacks during November 2015.

Plot of tweets by day of the week. 
For Muslims, Friday is the holiest and most virtuous day of the week, where they congregate for prayer. Traditionally, there are also five rounds of prayers. The increased activity on this day might point to religiosity towards Islam, although one cannot discount the fact that there are other religions where Friday has religious significance.


```python
weekday = tweets1['time'].groupby(tweets1['time'].map(lambda x: x.weekday())).agg('size')
weekday.index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_plot = weekday.plot(figsize=(12,4))
weekday_plot.set_ylabel('Number of tweets')
plt.show()
```


![png](/assets/ISIS-Project-1/output_32_0.jpg){:class='img-responsive'}


```python
hour = tweets1['time'].groupby(tweets1['time'].map(lambda x: x.hour)).agg('size')
hour_plot = hour.plot(figsize=(12,4))
hour_plot.set_ylabel('Number of tweets')
plt.show()
```


![png](/assets/ISIS-Project-1/output_33_0.jpg){:class='img-responsive'}


Plot of tweets by time of the day. Expected the different timezones to balance themselves out but strangely there seems to be some cycle here as if most of the users are from the similar timezones.

Below, we can see that some users are a lot more active than others.


```python
plt.figure(figsize=(25,7))

numberof = sns.barplot(x=no_tweets.index,y=no_tweets, color="#30a2da")
plt.ylabel('Number of tweets')

for item in numberof.get_xticklabels():
    item.set_rotation(30)
#no_tweets.set_xticklabels(rotation=90)

plt.show()
```


![png](/assets/ISIS-Project-1/output_36_0.jpg){:class='img-responsive'}



```python
tophash = hashcount.sort_values(ascending=False).head(25)
thplot = sns.barplot(x=tophash.index, y=tophash.values, color="#30a2da")

for item in thplot.get_xticklabels(): item.set_rotation(90)
plt.show()
```


![png](/assets/ISIS-Project-1/output_37_0.png){:class='img-responsive'}


Let's do a wordcloud. This should be rather similar to the barplot shown above, but might tell us more info about the content of tweets.


```python
wordcloud = WordCloud(relative_scaling=1, max_font_size=500, width=1800, height=1400, stopwords=sw).generate(text1)

interpolation='bilinear')
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


![png](/assets/ISIS-Project-1/output_39_0.png){:class='img-responsive'}


Some interesting notes on the words that appear above: 
- Fallujah (city in Iraq. Battle of Fallujah to take back city from ISIS on June 2017.)
- Dawlah (cycle, period, time of rule / period of success
- Mujahideen (people engaged in Jihad)
- Ramadi (city in central Irag)
- AmaqAgency (news agency linked to ISIS)
- Kuffar (derogatory, referring to non-believers)

Most of the tweets seem focused about current events happening in the Middle East, as well as reports on ISIS's activities and attacks there.

---
## Network Analysis

The network graph I need for this actually had more than a few thousand nodes, which Python was not able to handle. Some preprocessing was done to parse the dataset into an edgelist of a directed graph. The graphs visualization was then done on a software called Gephi.

Cleaning up the 'mentions' column to remove extra characters, and to preprocess data for loading into an edge list.


```python
a = []
b = ['']
tweets1['mentions'] = [a if x==b else x for x in tweets1['mentions']]
```

Prepare a dataframe as an edge list. Each node is a tweeter user, and each directed edge represents a mention (or retweet) to another user. Edge weights are the number of mentions.

A -> B : A mentioned B in tweet


```python
if tweets1['mentions'][0]:
    print ('a')
```

Function below flattens a nested list of strings. Similar to np.flatten, but for strings!


```python
def flatten_string(user_list):
    return[user.replace(' ','') for y in user_list if y for user in y]
```


```python
nl = tweets.groupby('username')['mentions'].apply(flatten_string)
# nl was later saved as a pickle called 'nodelist'.
```


```python
nodelist2 = nodelist.copy()
```


```python
nodelist2.head()
```




    username
    04_8_1437      [4_8_1437, 4_8_1437, 4_8_1437, xxxzzz333, 04_8...
    06230550_IS    [mustafaklash37, NikoRe_, kazabalanka2, scent_...
    1515Ummah      [BintLucy, bint_dutches14, lIlIlllIlIllI, lIlI...
    1Dawlah_III                                                   []
    432Mryam       [nusayba4a, dieinurage30_, _blvrnasiha, _blvrn...
    Name: mentions, dtype: object




```python
node1, node2, mentions = ([] for i in range(3))

for i in range(len(nodelist2)):

    for user, weight in Counter(nodelist2.values[i]).items():
        node1.append(nodelist2.index[i])
        node2.append(user)
        mentions.append(weight)
```


```python
edgelist = pd.DataFrame({'node1': node1,
                        'node2': node2,
                        'mentions': mentions})
edgelist = edgelist.loc[:,['node1', 'node2', 'mentions']]
edgelist.head()
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
      <th>node1</th>
      <th>node2</th>
      <th>mentions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>04_8_1437</td>
      <td>4_8_1437</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04_8_1437</td>
      <td>uhhgyfergbnj</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04_8_1437</td>
      <td>04_8_1437</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04_8_1437</td>
      <td>xxxzzz333</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06230550_IS</td>
      <td>TRENDING_WNEWS</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The directed nodes are shown above, with the number of mentions between users added in, if needed later.


```python
import networkx as nx

g = nx.DiGraph()
```

The above creates a digraph using graph G's connections. Note that doing 'gd=nx.Digraph(g)', where g is undirected, will cause duplicate edges, which makes the visualization inaccurate.


```python
for i, el in edgelist.iterrows():
    # i is the index, el is the values
    g.add_edge(el[0], el[1], mentions = float(el[2]))
```

Here, save the edge list into a file format that Gephi can read. Gephi is then used to process and visualize the graph.


```python
nx.write_gexf(g, 'mentions.gexf')
```

Note that values saved to g's edges must be integers or floats. The gexf format cannot handle float64 data types. 

### Network Diagrams from Gephi

Screenshots of network visualization done in Gephi. Influential users identified: Uncle_SamCoco, warrnews, mobi_ayubi, MaghrabiArabi, WarReporter1,  RamiAlLolah.

![png](/assets/ISIS-Project-1/output_67_0.png){:class='img-responsive'}

Notes on the graph: 
- Size of nodes determined by betweeness centrality of node
- Lighter shade represents higher indegree (more mentions).

Notice that some nodes are big in size, but darker in color. These are nodes who receive less mentions. However, their higher betweeness centrality indicates that they were key in information spreading through the network, in the form of shorter edge connections with more nodes. 

There is some activity going on in the center of the network - these nodes are lighter in shade and slightly larger in size than surrounding nodes. Shown below, these are where interactions happen between different communities.

The outer periphery of the network show less activity. These are likely other twitter users who were mentioned, but whose tweet activity were not captured by our dataset. 


![png](/assets/ISIS-Project-1/output_69_0.png){:class='img-responsive'}


![png](/assets/ISIS-Project-1/output_71_0.png){:class='img-responsive'}


Notes on the graph above: 
- colour shows the different communities in the network
- communities are nodes that are more highly connected (versus nodes that are not in the community)

One sees that are local communities surrounding the more influential users, with more interaction between different communities at the center of the network. This is verified when referring to the previous graphs diagram in blue, where the center of the network is lighter in colour, indicating more indegree activity (mentions between users).

The network analysis statistics are shown below. The node centrality statistic used is the betweenness centrality. 

The betweenness centrality of a graph node is first calculated thus: 


![png](/assets/ISIS-Project-1/output_73_0.png){:height="50%" width="50%"}


for a node v. The numerator is the number of shortest paths from node s to node t, passing through v, while the denominator is the total shortest paths passing through node s to node t. This is normalised through the following equation: 


![png](/assets/ISIS-Project-1/output_77_0.png){:height="50%" width="50%"}


This then gives a measure of the centrality of a node in the network. Nodes with higher betweenness centrality are more influential, as more information passes through that node. In this case, this would be influential twitter users who are key to information spreading. 


```python
bc = nets.sort_values(by=['betweenesscentrality'], ascending=False).head(10)
bcp = sns.barplot(bc.Id, bc.betweenesscentrality, color="#30a2da")
for item in bcp.get_xticklabels(): item.set_rotation(80)
plt.show()
# in this case, as each edge represents a 'mention', nodes with high betweeness centrality are key nodes for
# spread of information in this network. 
```



![png](/assets/ISIS-Project-1/output_65_0.png){:class='img-responsive'}



```python
# filtering for only users whose tweet activity was captured in this dataset. 
# if these values are not filtered, the outdegree values would be highly skewed towards 0.
netsf = nets[nets.isin(nodelist.index)['Id']]
iof = sns.jointplot(x='indegree',y='outdegree',data=netsf)
```


![png](/assets/ISIS-Project-1/output_66_0.png){:class='img-responsive'}


Even after filtering, the outdegree range is much larger than the indegree range. These users are broadcasting content (in the form of mentioning other users with words or video content) to users who were not in this dataset. 

### Assessing the feasibility of attack prediction

An existing paper (Pro-ISIS Fanboys Network Analysis and Attack Detection thorugh Twitter Data, Zhou, 2017) used an LSTM for attack prediction based on tweets frequency of influential users and hashtag frequency. Information used for prediction is high-frequency hashtags, influential users' tweets involvement, as well as the total tweets number for one day.

However, after looking at the dataset in more detail, I am not sure if attack prediction makes sense given the intent of the tweets - most of it looks informative in nature with links to news articles or comments on Middle East current events. 

Features I originally wanted to include:
- Number of hashtag mentions 
- Tweet behaviour of influential users (Uncle_SamCoco, RamiAIlolah, WarReporter1): number of mentions
- Day on which the attack occurs
- Violent sentiment rating in tweets content

In particular, violent sentiment rating is dropped because only a smaller proportion of the tweets are opinions. I would expect to get a skewed result for violent sentiment rating on tweets about news, which contains words like 'killed' or 'explosion'. 

---

### Transitioning to classification of tweet intent

I originally wanted to predict ISIS attacks with this data, but this was deemed unfeasible as the tweets' content were mainly centered on commentary on Middle East news, as well as sharing of propaganda. However, I thought it would be interesting to predict the intent of tweets based on the following categories: 

-  N: News. Informative in nature with statements on curent events or links to news.
- P: Propaganda. Links to ISIS news sources, or biased links.
- O: Opinion. Containing personal thoughts. Religious quotes were placed here as well as they constitute religious opinion.

#### Finding commonly occuring n-grams below. 

I later learned that there are NLP libraries that extract n-grams. However, it was a good exercise writing these functions myself. It did not take long to run, as the size of the dataset was not huge, and tweets as a rule only contained a maximum of 170 characters.


```python
# untokenize words. eg: ['ENGLISH','TRANSLATION'] to 'ENGLISH TRANSLATION'
def join_words(tokens):
    for i in range(len(tokens)):
        if i==0: x = tokens[i]
        else: x += ' '+tokens[i]
    return x
```


```python
def find_ngrams(text, n):
    text2list = RegexpTokenizer('\w+').tokenize(text)
    listofwords = [text2list[i:(i+n)] for i in range(len(text2list))]
    joinedwords = [join_words(words) for words in listofwords]
    return joinedwords
```


```python
grams2_content = [find_ngrams(day_tweets, 2) for day_tweets in content_date]
content['grams2'] = grams2_content
```

The functions written above were then run to find the commonly occuring n-grams, shown below.


```python
pd.Series(flatten_string(grams3_content)).value_counts().sort_values(ascending=False).head(10)
```




    https t RT           156
    https t co           137
    Iraq i army          110
    https t c            109
    the city of           99
    the Islamic State     92
    in the city           87
    Assad s army          84
    t co RT               68
    The Islamic State     67
    dtype: int64




```python
pd.Series(flatten_string(grams4_content)).value_counts().sort_values(ascending=False).head(10)
```




    in the city of                   69
    https t co RT                    67
    https t c RT                     59
    1 2 2 2                          50
    for the sake of                  26
    RT Did you know                  24
    ISIS claims responsibility of    18
    of the Islamic State             17
    Website Status Hacked by         17
    Syria https t RT                 17
    dtype: int64




```python
pd.Series(flatten_string(grams5_content)).value_counts().sort_values(ascending=False).head(20)
```




    CyberNews Website Status Hacked by       17
    for the sake of Allah                    15
    1 2 2 2 Caliphate_News                   13
    Sinai IED explosion targeted Egyptian    11
    files justpaste d278 a10870529 p         11
    came from thankSAll Who were             10
    this week came from thankSAll            10
    glimpse at the work of                   10
    A glimpse at the work                    10
    from thankSAll Who were yours            10
    and more last week See                   10
    best RTs this week came                  10
    RTs this week came from                  10
    s your audience growing via              10
    How s your audience growing              10
    in the city of Sirte                     10
    week came from thankSAll Who             10
    IED explosion targeted Egyptian army     10
    Destroyed 3 Turkish Army Tanks           10
    أربع ثكنات للروافض على طريق               9
    dtype: int64



### Manually label train/test instances: N (unbiased news sources), P (Propaganda) or O (Opinion). 

**Definitions**
- N: Unbiased news sources like Reuters, BBC, or factual reports on current events. Does not include information from ISIS (Amaq Agency)
- P: Biased news sources from ISIS, ot other news outlets. 
- O: Opinion. Individual opinion. 

As seen from commonly occuring ngrams above, these labelled training instances cannot be tweet duplicates (in the form of retweets), as this means information would bleed to the test set. 

To start with, the tweets have to be filtered only for EN (English), duplicates removed, then shuffled, sampled and labelled. Let's aim to label about 200-300 tweets first. I will use the 5-grams extracted to filter for retweets or tweet duplicates. This is a set of unique phrases in tweets (consisting of 5 words)


```python
grams5 = pd.Series(flatten_string(grams5_content)).value_counts().sort_values(ascending=False)
grams5RT = grams5[grams5>1]
```


```python
index2drop = []
for i in tweets2['tweets'].index:
    print (i)
    for word in set(grams5RT.index):
        flag=0
        if (word in tweets2['tweets'].iat[i])&(flag==0): 
            flag=1
            pass
        if (word in tweets2['tweets'].iat[i])&(flag==1): index2drop.append(i)
```


```python
tweets_uniq = tweets2.drop(tweets2.index[index2drop])
```

Tweets are then verified again and filtered for duplicates. 


```python
# then verify if there are duplicates in tweets2 by searching for top 5-grams.

content_uniq = tweets_uniq.groupby(tweets_uniq.time.dt.date)['tweets'].apply(lambda x: x.sum())
grams5_uniq = [find_ngrams(day_tweets, 5) for day_tweets in content_uniq]

pd.Series(flatten_string(grams5_uniq)).value_counts().sort_values(ascending=False)
```

Tweets are then filtered for those only in English. 


```python
# filter for tweets only in English
entweets_uniq = tweets_uniq[tweets_uniq['lang']=='en']
entweets_uniq_subset = entweets_uniq.sample(frac=0.5)
```
