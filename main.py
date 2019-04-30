import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from feature_extraction import word_ngrams, char_ngrams, pos_ngrams

data = pd.read_pickle('data/scaled_features.pkl')

male_count = len(data[data.Gender == 'M'].index)
female_count = len(data[data.Gender == 'F'].index)
frac = female_count / male_count

data_new = data.copy()
data_new = data_new.drop(data_new[data_new.Gender == 'M'].sample(frac=1 - frac).index)
data_new['Blog'] = data_new['Blog'].values.astype(str)

data_train_val, data_test, gender_train_val, gender_test = train_test_split(
    data_new.loc[:, data_new.columns != 'Gender'], data_new.Gender, test_size=0.1, shuffle=True,
    stratify=data_new.Gender, random_state=42)

data_train, data_val, gender_train, gender_val = train_test_split(data_train_val, gender_train_val, shuffle=True,
                                                                  test_size=0.22, stratify=gender_train_val,
                                                                  random_state=42)
data_folds = np.array_split(data_train_val, 10)
gender_folds = np.array_split(gender_train_val, 10)

ngram_range = (2, 8)
binary = True
count = 50000
scores = []
for i in range(10):
    start_time = datetime.now()
    test_data = data_folds[i]
    test_gender = gender_folds[i]

    train_data = pd.concat(data for data in data_folds[:i] + data_folds[i + 1:])
    train_gender = pd.concat(gender for gender in gender_folds[:i] + gender_folds[i + 1:])

    cv_word, transformed_words_train = word_ngrams(train_data, ngram_range=ngram_range, binary=binary)
    cv_char, transformed_char_train = char_ngrams(train_data, ngram_range=ngram_range, binary=binary)
    cv_pos, transformed_pos_train = pos_ngrams(train_data, ngram_range=ngram_range, binary=binary)
    train_data = train_data.drop(columns=['Blog', 'POS'])
    transformed_words_test = cv_word.transform(test_data.Blog)
    transformed_char_test = cv_char.transform(test_data.Blog)
    transformed_pos_test = cv_pos.transform(test_data.POS)
    test_data = test_data.drop(columns=['Blog', 'POS'])
    transformed_train = hstack((transformed_words_train, transformed_char_train, transformed_pos_train, train_data),
                               format='csr')
    transformed_test = hstack((transformed_words_test, transformed_char_test, transformed_pos_test, test_data),
                              format='csr')
    selector = GenericUnivariateSelect(chi2, 'k_best', param=count)
    selected_train = selector.fit_transform(transformed_train, train_gender)
    selected_test = selector.transform(transformed_test)

    clf = MLPClassifier(hidden_layer_sizes=(75, 75), solver='adam', activation='identity', early_stopping=True,
                        max_iter=2500, random_state=42)
    clf.fit(selected_train, train_gender)
    score = clf.score(selected_test, test_gender)
    scores.append(score)
    print("Score for iteration ", i, " = ", score)
    print("Time taken = ", datetime.now() - start_time)
    print("Params = ", clf.get_params())

print("10-fold CV scores = ", scores)
print("Mean CV score = ", np.mean(scores))

cv_word, transformed_words_train = word_ngrams(data_train, ngram_range=ngram_range, binary=binary)
cv_char, transformed_char_train = char_ngrams(data_train, ngram_range=ngram_range, binary=binary)
cv_pos, transformed_pos_train = pos_ngrams(data_train, ngram_range=ngram_range, binary=binary)
data_train = data_train.drop(columns=['Blog', 'POS'])
transformed_words_test = cv_word.transform(data_test.Blog)
transformed_char_test = cv_char.transform(data_test.Blog)
transformed_pos_test = cv_pos.transform(data_test.POS)
data_test = data_test.drop(columns=['Blog', 'POS'])

transformed_train = hstack((transformed_words_train, transformed_char_train, transformed_pos_train, data_train),
                           format='csr')
transformed_test = hstack((transformed_words_test, transformed_char_test, transformed_pos_test, data_test),
                          format='csr')
selector = GenericUnivariateSelect(chi2, 'k_best', param=count)
selected_train = selector.fit_transform(transformed_train, gender_train)
selected_test = selector.transform(transformed_test)

clf = MLPClassifier(hidden_layer_sizes=(75, 75), solver='adam', activation='identity', early_stopping=True,
                    max_iter=2500, random_state=42)
clf.fit(selected_train, gender_train)
print("Score on test data = ", clf.score(selected_test, gender_test))
print("Params = ", clf.get_params())
