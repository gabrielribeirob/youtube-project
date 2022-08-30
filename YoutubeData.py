# data/Videos
# t5/inglês/português
# aprimorar a coleta de perguntas;
# Montar um dataframe com pergunta / contexto (topic modelling --> armazenar um conjunto de palavras chaves);
# Isolar os tópicos por videos;
# O objetivo é ter um DF com perguntas e contexto;
# Reconhecimento de entidades talvez colocar no modelling;
# Squad banco de dados versão em portugues e versão 2.0;
# Rotular cada comentário
# As respostas não precisam ser direcionadas apenas para para perguntas;
# Ok, é interessante responder perguntas, mas comentar comentários pode ser legal também;
# Ou iremos criar um modelo baseado em banco de dados específico que no caso montaremos, ou usar um modelo pré-treinado, a ala BERT;
# No segundo caso só fazer um ajuste fino; 
from collections    import Counter
from datetime       import datetime, timedelta
from gensim         import corpora
from gensim.models  import LdaMulticore
from nltk.corpus    import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize
from pyLDAvis       import enable_notebook, gensim_models, save_html
from random         import randint
from time           import sleep
from transformers   import pipeline
from wordcloud      import WordCloud
import json, os, pickle, re, gensim
import pandas as pd

class YoutubeData():
  def __init__(self):
    self.data         = self.load_files()
    self.comments     = self.load_all_comments()
    self.descriptions = self.load_all_descriptions()
    self.authors      = self.get_authors()
    self.genre        = self.get_genre()
    self.set_data_per_channel()
  
  
  @staticmethod
  def _get_path_downloaded_files():
    root = 'Data/Videos/'
    paths = [os.path.join(root, i) for i in os.listdir(root)]
    return paths
  
  
  @staticmethod
  def ytdt2dt(ytdt:str):
    return int(re.sub(r'\D','',ytdt))
  
  
  @staticmethod
  def get_video_general_sentiment(video:dict) -> list[tuple]:
    labels = []
    thread = video['comments thread']
    if len(thread) == 1:
      return labels
    for i in thread:
      labels.append(i['sentiment'][0]['label'])
    labels  = Counter(labels).most_common()
    total   = sum([i[1] for i in labels])
    labels  = {i[0]:r2(i[1]/total) for i in labels}
    return labels
  
  def load_files(self):
    files = []
    for i in self._get_path_downloaded_files():
      d = json.load(open(i))
      d['duration']         = self.ytdt2dt(d['duration'])
      d['public sentiment'] = self.get_video_general_sentiment(d)
      d['gathered_at']      = datetime.fromtimestamp(d['gathered_at'])
      d['uploadDate']       = datetime.strptime(d['uploadDate'], '%Y-%m-%d')
      files.append(d)
    return files
  
  def load_all_comments(self):
    comments = [
      i['comments thread'] for i in self.data if len(i['comments thread']) > 0]
    comments = [j['text'].lower() for i in comments for j in i]
    return comments
  
  def load_all_descriptions(self):
    return [i['description'] for i in self.data]   
  
  def get_authors(self):
    return Counter(i['author'] for i in self.data).most_common()
 
  
  def get_genre(self):
    return Counter(i['genre'] for i in self.data).most_common()
  
  def group_by_channel(self):
    d = {k[0]:[] for k in self.authors}
    for i in self.data:
      d[i['author']].append(i)
    self.data_per_channel = d
 
  #----------------------------------------------------------------------------
  # SET DATA PER CHANNEL GROUP
  @staticmethod
  def _get_commentators(channel_data:list) -> set:
    commentators = set()
    for video in channel_data:
      thread = video['comments thread']
      c = set([i['author'] for i in thread])
      for i in c: commentators.add(i)
    return commentators
 
  @staticmethod
  def _get_public_sentiment_ix(channel_data:list) -> dict:
    d = {'Positive':0,'Negative':0,'Neutral':0}
    for video in channel_data:
      sent = video['public sentiment']
      if len(sent) == 0: pass
      else:
        for k,v in sent.items():
          d[k] += v
    d = {k:r2(v/max(d.values())) for k,v in d.items()}
    return d
 
  @staticmethod
  def _get_main_genre(channel_data:list) -> dict:
    d = [i['genre'] for i in channel_data]
    d = dict(Counter(d).most_common())
    d = {k:r2(v/sum(d.values())) for k,v in d.items()}
    return d
  @staticmethod
  def _get_creator_reactions(channel_data:list) -> int:
    d = 0
    for video in channel_data:
      thread = video['comments thread']
      if len(thread) > 0:
        for i in thread:
          d += 1 if i['creator_liked'] is True else 0
    return d
 
  @staticmethod
  def _get_post_interval(channel_data:list):
    uploads = [i['uploadDate'] for i in channel_data]
    # uploads = [datetime.strptime(i, '%Y-%m-%d') for i in uploads]
    uploads.sort()
    a,b = uploads[1:], uploads[:-1]
    intervals = []
    for i,j in zip(b,a):
      interval = j - i
      intervals.append(interval)
    intervals = [i.days for i in intervals]
    return intervals
 
  def set_data_per_channel(self, list_name:bool = False):
    if hasattr(self, 'data_per_channel') is False:
      self.group_by_channel()
    d = {}
    for k,v in self.data_per_channel.items():
      try: 
        _avg  = lambda x: r2(sum([int(i[x]) for i in v])/len(v))
        _sum  = lambda x: sum([int(i[x]) for i in v])
        comm  = self._get_commentators(v)
        _comm = lambda: comm
        intervals = self._get_post_interval(v)
        d[k]  = {}
        d[k]['Views: Avg']              = _avg('interactionCount')
        d[k]['Likes: Total']            = _sum('likes')
        d[k]['Likes: Avg']              = _avg('likes')
        d[k]['Duration: Avg (s)']       = _avg('duration')
        d[k]['Duration: Avg (str)']     = str(timedelta(seconds=_avg('duration')))
        d[k]['Comments: Total']         = _sum('comments')
        d[k]['Comments: Avg']           = _avg('comments')
        d[k]['Likes per Comments']      = r2(d[k]['Likes: Total']/_sum('comments'))
        d[k]['Main Genre']              = self._get_main_genre(v)
        d[k]['Commentators: Unique']    = len(comm)
        d[k]['Commentators: Avg']       = r2(len(comm)/len(v))
        d[k]['Commentators: Names']     = comm if list_name is True else _comm
        d[k]['Creator Reaction: Total'] = self._get_creator_reactions(v)
        d[k]['Creator Reaction: Avg']   = r2(d[k]['Creator Reaction: Total']/len(v))
        d[k]['Public Sentiment: Index'] = self._get_public_sentiment_ix(v)
        d[k]['Post Interval: Avg']      = r2(sum(intervals) / len(v))
        d[k]['Post Interval: Max']      = max(intervals)
      except Exception as e:
        print(e)
      # NLP
    self.channel_summary = d
    return d
  #----------------------------------------------------------------------------
  # FILTERS
  def filter_videos_by_most_comments(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['comments'] >= c[i['author']]['Comments: Avg']]
 
  def filter_videos_by_most_likes(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['likes'] >= c[i['author']]['Likes: Avg']] 
  def filter_videos_by_most_views(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['views'] >= c[i['author']]['Views: Avg']] 
  def filter_videos_by_least_comments(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['comments'] <= c[i['author']]['Comments: Avg']]
  def filter_videos_by_least_likes(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['likes'] <= c[i['author']]['Likes: Avg']]
  def filter_videos_by_least_views(self) -> list:
    d,c = self.data, self.channel_summary
    return [i for i in d if i['views'] <= c[i['author']]['Views: Avg']]
  def filter_videos_by(self, by:str, criteria:str='most') -> list:
    by = f'{criteria.lower()} {by.lower()}'
    if by   == 'most comments': return self.filter_videos_by_most_comments()
    elif by == 'most likes':    return self.filter_videos_by_most_likes()
    elif by == 'most views':    return self.filter_videos_by_most_views()
    elif by == 'least comments':return self.filter_videos_by_least_comments()
    elif by == 'least likes':   return self.filter_videos_by_least_likes()
    elif by == 'least views':   return self.filter_videos_by_least_views()
    else:   return []
  # ----------------------------------------------------------------------------
  # COMMENTS CONTENTS
  def find_tickers(self) -> list:
    tickers = []
    for i in self.comments:
      matches = re.findall(r'\b[A-z]{4}\d\d?\b', i, re.I)
      tickers.extend(matches)
    tickers = dict(Counter([i.lower() for i in tickers]).most_common())
    return tickers
 
  #----------------------------------------------------------------------------
  def to_dataframe(self):
    for k in self.channel_summary.keys():
      try:
        self.channel_summary[k]['Sentiment: Positive'] = self.channel_summary[k][
          'Public Sentiment: Index']['Positive']
        self.channel_summary[k]['Sentiment: Neutral'] = self.channel_summary[k][
          'Public Sentiment: Index']['Neutral']
        self.channel_summary[k]['Sentiment: Negative'] = self.channel_summary[k][
          'Public Sentiment: Index']['Negative']
        self.channel_summary[k].pop('Public Sentiment: Index')
        self.channel_summary[k].pop('Main Genre')
        self.channel_summary[k].pop('Commentators: Names')
      except Exception as e:
        print(e)
    return DataFrame(self.channel_summary)