from flair.models import SequenceTagger
from flair.data import Sentence
from string import punctuation

from gensim.models import Word2Vec
from nltk import sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd


def pos_tag(data, model_type='pos-fast'):
    tagger = SequenceTagger.load(model_type)
    blogs = data.Blog.astype(str).values
    blog_tags = []
    for blog in blogs:
        sentence = Sentence(blog)
        tagger.predict(sentence)
        tag_list = []
        for token in sentence.tokens:
            tag_list.append(token.tags['pos'].value)
        blog_tags.append(tag_list)
    data['POS'] = blog_tags
    return data


def f_measure(data):
    noun_set = {'NN', 'NNS', 'NNP', 'NNPS'}
    adjective_set = {'JJ', 'JJR', 'JJS', 'WDT'}
    preposition_set = {'IN'}
    article_set = {'DET', 'DT'}
    pronoun_set = {'PRP', 'PRP$', 'WP', 'WP$'}
    adverb_set = {'RB', 'RBR', 'RBS', 'EX'}
    interjection_set = {'UH'}
    verb_set = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'}

    pos = data.POS.values.astype(str)
    f_measure_list = []

    for tags in pos:
        fnoun = fadj = fprep = fart = fpron = fverb = fadv = fint = 0
        for tag in tags.split():
            if tag in noun_set:
                fnoun += 1
            elif tag in adjective_set:
                fadj += 1
            elif tag in preposition_set:
                fprep += 1
            elif tag in article_set:
                fart += 1
            elif tag in pronoun_set:
                fpron += 1
            elif tag in adverb_set:
                fadv += 1
            elif tag in interjection_set:
                fint += 1
            elif tag in verb_set:
                fverb += 1

        f = 0.5 * ((fnoun + fadj + fprep + fart) - (fpron + fverb + fadv + fint) + 100)
        f_measure_list.append(f)

    data['FMeasure'] = f_measure_list

    return data


def gender_preferential_features(data):
    blogs = data.Blog.astype(str).values
    fl = [[], [], [], [], [], [], [], [], [], [], []]
    for blog in blogs:
        tokens = str(blog).lower().split()
        f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = f10 = f11 = 0
        for token in tokens:
            if token.endswith('able'):
                f1 += 1
            elif token.endswith('al'):
                f2 += 1
            elif token.endswith('ful'):
                f3 += 1
            elif token.endswith('ible'):
                f4 += 1
            elif token.endswith('ic'):
                f5 += 1
            elif token.endswith('ive'):
                f6 += 1
            elif token.endswith('less'):
                f7 += 1
            elif token.endswith('ly'):
                f8 += 1
            elif token.endswith('ous'):
                f9 += 1
            elif 'sorry' in token:
                f10 += 1
            elif token.startswith('apolog'):
                f11 += 1
        fl[0].append(f1)
        fl[1].append(f2)
        fl[2].append(f3)
        fl[3].append(f4)
        fl[4].append(f5)
        fl[5].append(f6)
        fl[6].append(f7)
        fl[7].append(f8)
        fl[8].append(f9)
        fl[9].append(f10)
        fl[10].append(f11)

    data['f1'] = fl[0]
    data['f2'] = fl[1]
    data['f3'] = fl[2]
    data['f4'] = fl[3]
    data['f5'] = fl[4]
    data['f6'] = fl[5]
    data['f7'] = fl[6]
    data['f8'] = fl[7]
    data['f9'] = fl[8]
    data['f10'] = fl[9]
    data['f11'] = fl[10]

    return data


def word_factors(data):
    blogs = data.Blog.astype(str).values
    awf = pd.read_csv('data/Argamon_Word_Factors.csv')
    mwf = pd.read_csv('data/Mukherjee_Word_factors.csv')

    awf_categories = awf['Factor']
    mwf_categories = mwf['Factor']

    for factor in awf_categories:
        fctr = []
        words = set(map(lambda word: word.replace(',', ''), awf.loc[awf.Factor == factor, 'Words'].item().split()))
        for blog in blogs:
            ctr = 0
            for token in str(blog).split():
                if token in words:
                    ctr += 1
            fctr.append(ctr)

        data[factor + 'Count'] = fctr

    for factor in mwf_categories:
        fctr = []
        words = set(map(lambda word: word.replace(',', ''), mwf.loc[mwf.Factor == factor, 'Words'].item().split()))
        for blog in blogs:
            ctr = 0
            for token in str(blog).split():
                if token in words:
                    ctr += 1
            fctr.append(ctr)

        data[factor + 'Count'] = fctr

    return data


def tf_punctuation(data):
    """compute and return the term frequencies for punctuations"""
    blogs = data.Blog.astype(str).values
    punct = set(punctuation)
    punct_ctr = []

    for blog in blogs:
        ctr = 0
        tokens = list(blog)
        for token in tokens:
            if token in punct:
                ctr += 1
        punct_ctr.append(ctr)

    data['PunctuationCount'] = punct_ctr

    return data


def tf_stopwords(data):
    """compute and return frequencies of stopwords"""
    blogs = data.Blog.astype(str).values
    stop = set(stopwords.words('english'))
    stop_ctr = []

    for blog in blogs:
        stop_count = 0
        tokens = str(blog).split()
        for token in tokens:
            if token in stop:
                stop_count += 1
        stop_ctr.append(stop_count)

    data['StopCount'] = stop_ctr

    return data


def n_sentences(data):
    """compute and return number of sentences"""

    blogs = data.Blog.astype(str).values
    nsents = []

    for blog in blogs:
        sentences = sent_tokenize(blog)
        nsents.append(len(sentences))

    data['SentenceCount'] = nsents

    return data


def avg_sentence_length(data):
    """compute and return average length of sentences"""

    blogs = data.Blog.astype(str).values
    sentlengths = []

    for blog in blogs:
        sentences = sent_tokenize(blog)
        sumsent = 0
        for sentence in sentences:
            sumsent += len(sentence)
        sentlengths.append(sumsent / len(sentences))

    data['AvgSentLength'] = sentlengths

    return data


def blog_length(data):
    """compute and return length of blogs"""
    blogs = data.Blog.astype(str).values
    lengths = []

    for blog in blogs:
        lengths.append(len(blog))

    data['BlogLength'] = lengths

    return data


def tf_proper_nouns(data):
    """compute and return counts of proper nouns"""

    pos = data.POS.values
    fctr = []

    for tags in pos:
        ctr = 0
        for tag in tags.split():
            if tag in ('NNP', 'NNPS'):
                ctr += 1
        fctr.append(ctr)

    data['ProperNounCount'] = fctr

    return data


def count_upper_chars(data):
    ul = []
    blogs = data.Blog.astype(str).values
    for blog in blogs:
        uctr = 0
        for token in list(blog):
            if token.isupper():
                uctr += 1
        ul.append(uctr)

    data['UpperCaseChars'] = ul

    return data


def count_upper_words(data):
    ul = []
    blogs = data.Blog.astype(str).values
    for blog in blogs:
        uctr = 0
        for token in blog.split():
            if token.isupper():
                uctr += 1
        ul.append(uctr)

    data['UpperCaseWords'] = ul

    return data


def count_title_words(data):
    ul = []
    blogs = data.Blog.astype(str).values
    for blog in blogs:
        uctr = 0
        for token in blog.split():
            if token.istitle():
                uctr += 1
        ul.append(uctr)

    data['TitleCaseWords'] = ul

    return data


def word_ngrams(data, ngram_range=(2, 9), binary=True):
    blogs = data.Blog.astype(str).values

    cv = CountVectorizer(ngram_range=ngram_range, binary=binary)
    transformed_blog = cv.fit_transform(blogs)

    data['TransformedBlog'] = transformed_blog

    return data


def char_ngrams(data, ngram_range=(2, 9), binary=True):
    blogs = data.Blog.astype(str).values

    cv = CountVectorizer(ngram_range=ngram_range, binary=binary, analyzer='char')
    transformed_blog = cv.fit_transform(blogs)

    data['TransformedBlog'] = transformed_blog

    return data


def pos_ngrams(data, ngram_range=(2, 9), binary=True):
    pos = data.POS.values

    cv = CountVectorizer(ngram_range=ngram_range, binary=binary)
    transformed_pos = cv.fit_transform(pos)

    data['TransformedBlog'] = transformed_pos

    return data


def word2vec(data, skip_gram=1):
    tt = TweetTokenizer()
    plist = set(list(punctuation))
    blogs = data.Blog.astype(str).apply(lambda blog: tt.tokenize(blog)).apply(
        lambda tokens: [token for token in tokens if token not in plist]).values
    model = Word2Vec(blogs, min_count=10, workers=4, sg=skip_gram)

    return model
