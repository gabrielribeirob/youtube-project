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
# from concurrent.futures import ProcessPoolExecutor
import json, os, pickle, re, gensim
import pandas as pd
class YoutubeDataLanguageToolkit():
  def __init__(self, language:str='portuguese'):
    self.stopwords = self._set_stopwords(language)

  @staticmethod
  def filter_sentences_per_nouns(sentences:list, lemma=True) -> list:
    """"""
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
      sentence = [i for i in words if i not in nltk.stopwords]
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
    id2word = corpora.Dictionary(sentences) # plural de corpus
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
    questions = [i for i in sentences if '?' in i] # olhar preposições
    questions = [self.isolate_question_and_context(i) for i in questions]
    self.questions = questions
    self.generate_wordcloud(questions) if wordcloud is True else None
    return questions
  


  def see_topics(self, sentences:list[str], topics=20):
    sents =   self.filter_sentences_per_nouns(sentences)
    lda   =   self.set_topic_model(sents, topics=topics)
    return lda

