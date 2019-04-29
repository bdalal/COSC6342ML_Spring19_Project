import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.neural_network import MLPClassifier

from feature_extraction import word_ngrams, char_ngrams, pos_ngrams

data = pd.read_csv('data/blog-gender-dataset_csv.csv')

male_count = len(data[data.Gender == 'M'].index)
female_count = len(data[data.Gender == 'F'].index)
frac = female_count / male_count

data_new = data.copy()
data_new = data_new.drop(data_new[data_new.Gender == 'M'].sample(frac=1 - frac).index)
data_new['Blog'] = data_new['Blog'].values.astype(str)

data_train_val, data_test, gender_train_val, gender_test = train_test_split(
    data_new.loc[:, data_new.columns != 'Gender'], data_new.Gender, test_size=0.1, shuffle=True,
    stratify=data_new.Gender)

data_train, data_val, gender_train, gender_val = train_test_split(data_train_val, gender_train_val, shuffle=True,
                                                                  test_size=0.22, stratify=gender_train_val)

ngram_range = (2, 9)
binary = True
count = 50000

cv_word, transformed_words_train = word_ngrams(data_train, ngram_range=ngram_range, binary=binary)
cv_char, transformed_char_train = char_ngrams(data_train, ngram_range=ngram_range, binary=binary)
cv_pos, transformed_pos_train = pos_ngrams(data_train, ngram_range=ngram_range, binary=binary)
transformed_words_val = cv_word.transform(data_val)
transformed_char_val = cv_char.transform(data_val)
transformed_pos_val = cv_pos.transform(data_val)
transformed_words_test = cv_word.transform(data_test)
transformed_char_test = cv_char.transform(data_test)
transformed_pos_test = cv_pos.transform(data_test)
transformed_train = hstack((transformed_words_train, transformed_char_train, transformed_pos_train), format='csr')
transformed_val = hstack((transformed_words_val, transformed_char_val, transformed_pos_val), format='csr')
transformed_test = hstack((transformed_words_test, transformed_char_test, transformed_pos_test), format='csr')
selector = GenericUnivariateSelect(chi2, 'k_best', param=count)
selected_train = selector.fit_transform(transformed_train, gender_train)
selected_val = selector.transform(transformed_val)
selected_test = selector.transform(transformed_test)

clf = MLPClassifier(hidden_layer_sizes=(75, 75), solver='adam', activation='identity', early_stopping=True,
                    max_iter=2500)
clf.fit(selected_train, gender_train)

