import gensim
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    target_names = ['review', 'sentiment']
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    print("Classification report:\n" + classification_report(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')


def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data['review'])
    predictions = classifier.predict(data_features)
    target = data['sentiment']
    evaluate_prediction(predictions, target)


def bag_of_words():
    count_vectorizer = CountVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=3000)

    train_data_features = count_vectorizer.fit_transform(train_data['review'])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['sentiment'])

    predict(count_vectorizer, logreg, test_data)


def n_grams():
    n_gram_vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=([2, 5]),
        tokenizer=None,
        preprocessor=None,
        max_features=3000)

    train_data_features = n_gram_vectorizer.fit_transform(train_data['review'])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['sentiment'])

    predict(n_gram_vectorizer, logreg, test_data)


def tf_idf():
    tf_vect = TfidfVectorizer(
        min_df=2, tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english')

    train_data_features = tf_vect.fit_transform(train_data['review'])

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['sentiment'])

    predict(tf_vect, logreg, test_data)


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        print("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(word)
    return tokens


def word2vec():
    wv = KeyedVectors.load_word2vec_format(
        'D:\\Study\\DLP_classification\\files\\GoogleNews-vectors-negative300.bin.gz',
        binary=True)
    wv.init_sims(replace=True)

    test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r['review']), axis=1).values
    train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r['review']), axis=1).values

    X_train_word_average = word_averaging_list(wv, train_tokenized)
    X_test_word_average = word_averaging_list(wv, test_tokenized)

    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)

    logreg = logreg.fit(X_train_word_average, train_data['sentiment'])
    predicted = logreg.predict(X_test_word_average)

    evaluate_prediction(predicted, test_data.tag)


# Import data
df = pd.read_csv('D:\\Study\\DLP_classification\\files\\movie_reviews.csv')
print(df['review'].apply(lambda x: len(x.split(' '))).sum())

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)

print(len(test_data))
#
# # Test Bag of Words
# print('\nBag of Words start!')
# start = time.time()
# bag_of_words()
# end = time.time()
# print('Bag of Words time: ' + str(end - start))
#
# # Test n-grams
# print('\nN-grams start!')
# start = time.time()
# n_grams()
# end = time.time()
# print('N-grams time: ' + str(end - start))
#
# # Test tf-idf
# print('\nTF-IDF start!')
# start = time.time()
# tf_idf()
# end = time.time()
# print('TF-IDF time: ' + str(end - start))

# Test word2vec
print('\nWord2vec start!')
start = time.time()
word2vec()
end = time.time()
print('Word2vec time: ' + str(end - start))

