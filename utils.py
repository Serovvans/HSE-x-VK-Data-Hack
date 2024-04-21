from transformers import T5ForConditionalGeneration, T5Tokenizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

import pymorphy3
morph = pymorphy3.MorphAnalyzer()


from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)

import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def clean_text(text):
    # Удаляем пунктуацию
    text = re.sub(r'[^\w\s]', '', text)
    # Приводим к нижнему регистру
    text = text.lower()
    # Разбиваем текст на слова
    words = text.split()
    # Удаляем стоп-слова
    stop_words = set(stopwords.words("russian"))
    filtered_words = [word for word in words if word not in stop_words]
    # Собираем текст из отфильтрованных слов
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

def get_names(text):
  names_extractor = NamesExtractor(morph_vocab)

  doc = Doc(text)
  doc.segment(segmenter)
  doc.tag_ner(ner_tagger)

  tags = []
  # Нормализация и вывод результатов
  for span in doc.spans:
      span.normalize(morph_vocab)
      # if span.type in {'PER', 'LOC', 'ORG'}:  # Имена, локации, организации
      word = morph.parse(span.normal)[0]
      tags.append(word.inflect({'nomn'}).word)

  # Например, можно отфильтровать и вывести только уникальные имена и места
  unique_entities = list(set(tags))

  return tags

def get_tags(text):
  names_extractor = NamesExtractor(morph_vocab)

  doc = Doc(text)
  doc.segment(segmenter)
  doc.tag_ner(ner_tagger)

  tags = []
  # Нормализация и вывод результатов
  for span in doc.spans:
      span.normalize(morph_vocab)
      if span.type in {'PER', 'LOC', 'ORG'}:  # Имена, локации, организации
        word = morph.parse(span.normal)[0]
        try:
          tags.append(word.inflect({'nomn'}).word)
        except:
          pass

  # Например, можно отфильтровать и вывести только уникальные имена и места
  unique_entities = list(set(tags))

  return tags

def get_popular_words(text_):
  doc = Doc(text_)
  doc.segment(segmenter)
  doc.tag_morph(morph_tagger)

  noun_counts = {}

  for token in doc.tokens:
    token.lemmatize(morph_vocab)
    if token.pos == 'NOUN':  # Фильтрация существительных
        lemma = token.lemma
        noun_counts[lemma] = noun_counts.get(lemma, 0) + 1

    sorted_nouns = list(sorted(noun_counts.items(), key=lambda item: item[1], reverse=True))
    list_ = sorted_nouns[:5]

  return set(list_)

def get_key_words(text, summarize):
  keys = list(map(lambda x: x[0], get_popular_words(text)))
  sobst = get_tags(summarize)
  return {
      "popular": keys,
      "names": sobst
  }


def extract_keywords_lda(text, num_topics=3, num_words=5):
    # Токенизация текста и удаление стоп-слов и пунктуации
    tokens = clean_text(text).split()

    # Удаляем короткие токены
    tokens = [token for token in tokens if len(token) > 2]

    # Создание словаря
    dictionary = corpora.Dictionary([tokens])

    # Создание корпуса
    corpus = [dictionary.doc2bow(tokens)]

    # Обучение модели LDA
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    # Извлечение ключевых слов из тем
    keywords = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=num_words)
        keywords.extend([word for word, _ in topic_words])

    return set(keywords)


labels_codes = {
    "Политика": 1,
    "Экономика": 2,
    "Общество": 3,
    "Закон и право": 4,
    "Кино": 5,
    "Телевидение": 6,
    "Персоны": 7,
    "События": 8,
    "Бренды": 9,
    "Наука": 10,
    "Гаджеты": 11,
    "Coцсети": 12,
    "Технологии": 13,
    "Опросы": 14,
    "Головоломки": 15,
    "Дом": 16,
    "Транспорт": 17,
    "Погода": 18,
    "Рецепты": 19,
    "Мода": 20,
    "Красота": 21,
    "Животные": 22
}


def summarize(text):
    tokenizer = T5Tokenizer.from_pretrained("sarahai/ruT5-base-summarizer")
    model = T5ForConditionalGeneration.from_pretrained("sarahai/ruT5-base-summarizer")

    model = model.to('cpu')

    # Применение ruT5 для суммирования текста
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs][0]
    
    return summary


def load_model_from_files(config_path="./model/config.json", model_path="./model/model.safetensors", num_labels=22, device='cpu'):
    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Загрузка конфигурации модели
    config = BertConfig.from_pretrained(config_path)
    
    # Загрузка модели BERT с предварительно обученными весами
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()  # Перевод модели в режим оценки (не тренировки)
    return tokenizer, model

def predict_class(text, tokenizer, model, device='cpu'):
    # Токенизация текста и добавление специальных токенов [CLS] и [SEP]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs.to(device)
    
    # Получение предсказаний модели
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Преобразование выходных значений в вероятности с помощью softmax
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # Выбор индекса класса с наибольшей вероятностью
    predicted_class = torch.argmax(logits, dim=1).item()
    
    for clas, key in labels_codes.items():
        if predicted_class == key:
            predicted_class = clas
            break
            
    
    return predicted_class
