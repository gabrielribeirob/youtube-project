from bs4            import BeautifulSoup
from collections    import Counter
from datetime       import datetime, timedelta
from gensim         import corpora
from gensim.models  import LdaMulticore
from math           import ceil
from nltk.corpus    import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize
from pandas         import DataFrame
from pyLDAvis       import enable_notebook, gensim_models, save_html
from random         import randint
from time           import sleep
from selenium       import webdriver
from transformers   import pipeline
from wordcloud      import WordCloud
from selenium.webdriver.common.keys     import Keys
from selenium.webdriver.common.by       import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from concurrent.futures import ProcessPoolExecutor
import  json, os, pickle, re, spacy

# !python -m spacy download pt_core_news_md
nlp = spacy.load('pt_core_news_md')
right_now = lambda: int(datetime.timestamp(datetime.now()))


BASE        = 'https://www.youtube.com'
BASE_SEARCH = 'https://www.youtube.com/results?search_query='

CT  = 'content-text'
CHB = 'creator-heart-button'
VCM = 'vote-count-middle'
PTT = 'published-time-text'
YFT = 'yt-formatted-string'
YIB = 'yt-icon-button'

r2  = lambda x: round(x,2)

# LANGUAGE MODELS
sentiment_analysis = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment', use_fast = False)
# finbert_br = pipeline('text-classification', model='turing-usp/FinBertPTBR')
# ner = pipeline("ner", "xlm-roberta-large-finetuned-conll03-english")


class YouTubeBot():
  def __init__(self):
    self.driver       = self.initialize_browser(url=BASE)
    self._SCROLLTIME  = 30
    
  @staticmethod
  def initialize_browser(url:str=BASE) -> webdriver:
    """Inicianliza o driver do selenium já encaminhado o browser para uma url específica

    Args:
        url (str, optional): URL para inicialização. Defaults to BASE (https://www.youtube.com).

    Returns:
        webdriver: abre o browser a partir do driver a acessa a url específica nos parâmetros
    """
    # ff_profile = webdriver.FirefoxProfile()
    options=Options()
    # don't load images
    options.set_preference('permissions.default.image', 2)
    options.set_preference('extensions.contentblocker.enabled', True)
    options.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', False)
    # no video autoplay
    options.set_preference('media.autoplay.default', 0)
    options.set_preference('media.autoplay.allow-muted', False)
    driver = webdriver.Firefox(options=options)
    # driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.get(url)
    return driver

  # DOWNLOADER ------------------------------
  def download_video_data(self, url:str):
    """Faz o download dos dados do vídeo em formato json salvadondo-os na pasta específicada na variável cache

    Args:
        url (str): url do vídeo que terá o dados extraídos
    """
    cache = [i.replace('.json','') for i in os.listdir('Data/Videos/')]
    if url not in cache:
      try:
        video = YouTubeVideos(url=url, bot=self.driver, thread=True)
        video.export()
      except Exception as e:
        print(e)

  def downloader(self, channel:str):
    """Cria uma a variável interna self.links contendo o link dos vídeos do canal.
       Após isso, passa link a link realizando a extração dos dados do vídeo apartir da função interna
       self.download_video_data.


    Args:
        channel (str): nome do canal que terá o vídeos extraídos
    """
    self.channel_name = channel
    self.open_channel(channel)
    original_window = self.driver.current_window_handle
    links = self.get_channel_videos_links()
    links = [i for i in links if 'shorts'  not in i]
    self.links = links
    for i in links:
      self.driver.switch_to.new_window('tab')
      self.download_video_data(i)
      self.driver.close()
      self.driver.switch_to.window(original_window)
      sleep(1)

  # -----------------------------------------
  def get_number_of_videos(self):
    """Pega o número de vídeos que o canal possui a partir da base de pesquisa, e salva o valor na variável interna
       self.nvideos.
       Além disse acessa inicializa o driver na url do self.videos_url que por sua vez será a url do canal em questão 
    """
    sleep(1)
    
    # self.driver.get(os.path.join(self.where_am_i(),'?view=57'))
    self.driver.get(BASE_SEARCH+self.channel_name)
    # self.driver.find_element(by=By.ID, value="play-button").click()
    # sleep(1)
    # tree = self.get_tree().find_all('a')
    # tree = [i['href'] for i in tree if 'href' in i.attrs]
    # playlist = [i for i in tree if 'playlist' in i][0]
    # sleep(1)
    # self.driver.get(BASE + playlist)
    # stats = self.get_tree().find('div',{'id':'stats'}).text.split('video')[0]
    # self.nvideos = int(re.sub(r'\D','',stats))
    element_present = EC.presence_of_element_located((By.ID, 'video-count'))
    WebDriverWait(self.driver, 10).until(element_present)
    # tree = BeautifulSoup(d.page_source,'lxml')
    tree = self.get_tree()
    video_count = tree.find('span', {'id': 'video-count'}).get_text()
    self.nvideos = int(re.sub(r'\D','',video_count))
    sleep(1)
    self.driver.get(self.videos_url)

  def get_page_source(self):
    """Pega a source da página em que o driver inicializou

    Returns:
        SwitchTo: um objeto contendo todas as opções para mudanças de página (https://selenium-python.readthedocs.io/api.html)
    """
    return self.driver.page_source

  def get_tree(self):
    """Pega a árvore html da página em questão

    Returns:
        BeautifulSoup Object: uma descrição do html da página
    """
    return BeautifulSoup(self.driver.page_source,'lxml')

  def open_channel(self, channel:str) -> webdriver:
    """Abre o canal específicado

    Args:
        channel (str): canal que será aberto

    Returns:
        webdriver: inicializa o driver acessando a url do canal
    """
    
    url = os.path.join(BASE, 'c', channel)
    self.driver.get(url)

  def open_channel_videos(self):
    """Abre a área de videos do canal. Seu objetivo princiapl é para depois coletar o
      o link de todos os videos
    """
    url = os.path.join(self.where_am_i(),'videos')
    self.videos_url = url
    self.driver.get(url)

  def scroll_down(self, n=30):
    """Realiza o scroll down da página 

    Args:ll
        n (int, optional): número de vezes que o comando send_keys será invocado. Defaults to 30.
    """
    if n < 10:
      for _ in range(n):
          # height = self.driver.execute_script("return document.body.scrollHeight")
          sleep(1)
          self.driver.find_element(by=By.TAG_NAME, value='body').send_keys(Keys.PAGE_DOWN)
          sleep(1)
    else:
      count = set()
      while len(count) <= self.nvideos:
        sleep(0.5)
        self.driver.find_element(by=By.TAG_NAME, value='body').send_keys(Keys.PAGE_DOWN)
        [count.add(i.get_attribute('href')) for i in self.driver.find_elements(By.TAG_NAME, value="a")]
        sleep(0.5)

        # body = self.driver.find_element(by=By.TAG_NAME, value='body')
        # for _ in range(n):
        #   # for _ in range(40): 
        #     self.driver.send_keys(Keys.COMMAND + Keys.ARROW_DOWN)
        #     # self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

        

  def get_channel_videos_links(self) -> list:
    """Cria uma lista contendo todos os links dos vídeos do canal em questão

    Returns:
        list: lista com o link dos videos do canal
    """
    self.open_channel_videos()
    self.get_number_of_videos()
    rollingtimes = ceil(self.nvideos/self._SCROLLTIME)
    self.scroll_down(rollingtimes)
    tree    = BeautifulSoup(self.driver.page_source,'lxml')
    videos  = tree.find_all('h3')
    links   = [i.a['href'] for i in videos if i.a is not None]
    links   = [i.replace('/','') for i in links]
    self.channel_videos = links
    fname = self.channel_name
    with open(f'cache/{fname}','w') as f: f.write(str(links))
    return links

  def quit(self):
    self.driver.quit()

  def where_am_i(self) -> str:
    return self.driver.current_url


class YouTubeVideos(YouTubeBot):
  def __init__(self, url, bot, thread=False):
    self.url    = os.path.join(BASE, url)
    if bot is False:
      self.driver = self.initialize_browser()
    else:
      self.driver = bot
    self.open_video()
    sleep(randint(4,8))
    self.info   = self.get_info_data()
    self.get_comments_thread() if thread is True else None
    self.info['comments thread'] = self.comments
  
  @staticmethod
  def _format_comment(c) -> dict:
    author = c.h3.text.strip()
    date   = c.find(YFT,{'class':PTT}).text.strip()
    text   = c.find(YFT,{'id':CT}).text.strip()
    likes  = c.find('span',{'id':VCM})
    likes  = int(re.sub(r'\D', '', likes['aria-label'])) if 'aria-label' in likes.attrs else 0
    creator_liked = False if c.find(YIB,{'id':CHB}) is None else True
    sentiment = sentiment_analysis(text)
    d = {'author':author, 'date':date, 'text':text, 'likes':likes, 'creator_liked':creator_liked, 'sentiment':sentiment}
    return d

  def open_video(self):
    self.driver.get(self.url)

  def export(self):
    fname = self.url.split('?')[-1]
    json.dump(self.info, open(f'Data/Videos/{fname}.json','w'))

  def get_likes(self) -> int:
    tree      = BeautifulSoup(self.get_page_source(), 'lxml')
    container = tree.find('div',{'id':'menu-container'})
    text      = container.find(YFT, {'id':'text'})
    likes     = text['aria-label'] if 'aria-label' in text.attrs else 0
    likes     = int(re.sub(r'\D','',likes))
    return likes

  def get_comments_number(self) -> int:
    self.scroll_down(1)
    return int(re.sub(r'\D','',self.get_tree().find('h2',{'id':'count'}).text))

  def get_comments_thread(self) -> list:
    n = self.get_comments_number()
    self.scroll_down(ceil(n/8)) 
    comments = []
    thread = self.get_tree().find_all("ytd-comment-thread-renderer")
    for i in thread:
      comments.append(self._format_comment(i))
    self.comments = comments
    return comments

  def get_info_data(self) -> dict:
    tree = BeautifulSoup(self.get_page_source(), 'lxml')
    data = json.loads(tree.find_all('script', {'id':'scriptTag'})[0].contents[0])
    data['likes']       = self.get_likes()
    data['comments']    = self.get_comments_number()
    data['gathered_at'] = datetime.now().timestamp()
    return data


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


class YoutubeDataLanguageToolkit():
  def __init__(self, language:str='portuguese'):
    self.stopwords = self._set_stopwords(language)

  @staticmethod
  def filter_sentences_per_nouns(sentences:list, lemma=True) -> list:
    sents = [nlp(i) for i in sentences]
    sents = [
      [j for j in i if j.pos_ in ['NOUN'] or re.match(r'\w{4}\d\d?',j.pos_, re.I)] for i in sents]
    sents = [[j.lemma_ if lemma is True else j.text for j in i] for i in sents]
    sents = [[j.lower() for j in i if len(j.strip()) > 1] for i in sents]
    sents = [i for i in sents if len(i) > 0]
    return sents

  @staticmethod
  def generate_wordcloud(sentences:list):
    long_string = ','.join(sentences)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    wordcloud.to_file(f'wc_{datetime.timestamp(datetime.now())}.png')

  @staticmethod
  def get_most_freq_words(self, sentences:list[list]) -> dict:
    bag = self.set_bag_of_tokens(sentences)
    return dict(Counter(bag).most_common())

  @staticmethod
  def prep_list_of_sentences(sentences):
    stopwords_ptb = stopwords.words('portuguese')
    stopwords_ptb.append('pra')
    sentences = [re.sub(r'[,\.!?]', '', i) for i in sentences]
    cleaned = [' '.join(
      [j for j in word_tokenize(i) if j.lower() not in stopwords_ptb]) for i in sentences]
    return cleaned

  @staticmethod
  def remove_ptb_special_chars(text:str) -> str:
    """Remove the special characters from a portuguese text, for example: "áéíóú" -> "aeiou"
    """
    text = text.lower()
    text = re.sub('[àãáâä]','a', text)
    text = re.sub('[èéêë]', 'e', text)
    text = re.sub('[ìíîï]', 'i', text)
    text = re.sub('[õóôö]', 'o', text)
    text = re.sub('[ùúûü]', 'u', text)
    text = re.sub('[ç]',    'c', text)
    text = re.sub('[ñ]',    'n', text)
    return text

  @staticmethod
  def remove_ptb_contractions(text:str) -> str:
    t = text.lower()
    t = re.sub(r'\bpq\b', 'porque', t)
    t = re.sub(r'\bq\b',  'que', t)
    t = re.sub(r'\bvc\b', 'voce', t)
    return t

  @staticmethod
  def remove_stopwords(sentences) -> list:
    sentences = sentences if type(sentences) is list else [sentences]
    cleaned = []
    for sentence in sentences:
      words = word_tokenize(sentence)
      sentence = [i for i in words if i not in stopwords]
      cleaned.append(' '.join(sentence))
    return cleaned

  @staticmethod
  def set_bag_of_tokens(list_of_sentences):
    bag = []
    for i in list_of_sentences:
      bag.extend(i)
    return bag

  @staticmethod
  def set_topic_model(sentences:list[list], topics=20, pickling=True, html=True):
    lda_path = f'./viz/topic_modelling/lda_{right_now()}'
    id2word = corpora.Dictionary(sentences)
    corpus  = [id2word.doc2bow(i) for i in sentences]
    lda     = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=topics)
    enable_notebook()
    viz_prepared = gensim_models.prepare(lda, corpus, id2word)
    # Controllers --------------------------------------------------------------
    if pickling is True:
      with open(lda_path, 'wb') as f:
        pickle.dump(viz_prepared, f)
    if html is True:
      save_html(viz_prepared, lda_path+'.html')
    return lda

  def _set_stopwords(self, language:str) -> list:
    if language == 'portuguese':
      stopwords_ptb = stopwords.words('portuguese')
      stopwords_ptb.extend(['pra', 'voce'])
      stopwords_ptb = [self.remove_ptb_special_chars(i) for i in stopwords_ptb]
      return stopwords_ptb
    else:
      return stopwords.words(language)

  def clean_text(self, text:str):
    t = text
    # change contractions
    t = self.remove_ptb_special_chars(t)
    t = self.remove_ptb_contractions(t)
    t = re.sub(r'\s+',      ' ',    t)
    t = re.sub(r' \?',      '?',    t)
    t = re.sub(r' !',       '!',    t)
    t = re.sub(r'^(\?!)',   '',     t)
    t = re.sub(r'\bq\b',    'que',  t)
    t = ' '.join([i for i in t.split() if i not in self.stopwords])
    return t

  def isolate_question_and_context(self, sentence:str) -> str:
    question = []
    i = sent_tokenize(sentence)
    for j in i:
      if '?' in j:
        ix = i.index(j)
        try:
          question.append(i[ix-1]) if i[ix-1] not in question else None
        except IndexError:
          pass
        question.append(j) if j not in question else None
    question = ' '.join(question)
    question = self.clean_text(question)
    return question

  def get_questions(self, sentences, wordcloud=False):
    questions = [i for i in sentences if '?' in i]
    questions = [self.isolate_question_and_context(i) for i in questions]
    self.questions = questions
    self.generate_wordcloud(questions) if wordcloud is True else None
    return questions

  def see_topics(self, sentences:list[str], topics=20):
    sents = self.filter_sentences_per_nouns(sentences)
    lda =   self.set_topic_model(sents, topics=topics)
    return lda


canais = [
  'Mepoupenaweb', 
  'ThiagoNigro', 
  'ClubedoValor',
  'RafaelSeabra',
  'GustavocerbasiBr',
  'EconoMirna',
  'JúliaMendonça',
  'PatriciaLages',
  'AcademiadoDinheiroOficial',
  'Euqueroinvestir',
  'investidorsardinha',
  'CanaldoHolder',
  'decoracasas',
  'iberethenorio',
  'welingtoninfo',
  'rezendeevil',
  'coisadenerd'
]

for c in canais:
  yt = YouTubeBot()
  yt.downloader(c)

yt   = YoutubeData()
com  = yt.comments
# ltk  = YoutubeDataLanguageToolkit()
# ltk.get_questions(com, wordcloud=False)
# sentences = ltk.questions
# ltk.see_topics(sentences, topics=40)


# https://www.youtube.com/watch?v=3hr2okL89oo
# https://www.youtube.com/watch?v=xUEHew8XjxY
# https://www.youtube.com/watch?v=SGzb1KrkwT8
# https://www.youtube.com/user/Claudiogo
