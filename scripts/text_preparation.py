import os
import nltk as nltk
import textract
import progressbar
from nltk.corpus import stopwords

# nltk.download()

INPUT_DIRECTORY = 'D:\\Study\\DLP_classification\\files\\input\\'
ORIGIN_DIRECTORY = 'D:\\Study\\DLP_classification\\files\\data\\origin\\'
CLEAN_DIRECTORY = 'D:\\Study\\DLP_classification\\files\\data\\clean\\'
EMBEDDINGS_DIRECTORY = 'D:\\Study\\DLP_classification\\files\\data\\clean\\embeddings\\'

my_stopwords = ['а', 'бы', 'без', 'был', 'быть', 'в', 'вас', 'вы', 'да', 'для', 'до', 'его', 'если', 'есть', 'еще',
                'ещё', 'ж', 'же', 'за', 'и', 'из', 'или', 'им', 'их', 'к', 'как', 'кем', 'когда', 'кого', 'которая',
                'которые', 'который', 'кто', 'ли', 'лишь', 'мы', 'на', 'над', 'нас', 'не', 'нет', 'несколько', 'ни',
                'но', 'о', 'об', 'он', 'они', 'от', 'при', 'по', 'под', 'пока', 'про', 'рф', 'с', 'свой', 'себя', 'со',
                'так', 'те', 'тем', 'тех', 'то', 'того', 'тот', 'ты', 'у', 'уже', 'чего', 'чем', 'через', 'что',
                'чтобы', 'эта', 'эти','это', 'этого', 'этой', 'я', 'com', 'be', 'html', 'http', 'https', 'm', 'my',
                'of', 'org', 'pic', 'ru', 's', 'stat', 'statu', 'status', 'tatus', 'twitter', 'youtu', 'youtube', 'ua',
                'www']


def import_text_from_documents():
    # get files list
    # extract text from each file with textract
    # save it to txt file
    input_files = os.listdir(INPUT_DIRECTORY)
    # bar = progressbar.ProgressBar(len(input_files)).start()
    for file in input_files:
        text_bytes = textract.process(INPUT_DIRECTORY + file, encoding='cp1251')
        new_file = os.path.splitext(file)[0] + '.txt'
        with open(ORIGIN_DIRECTORY + new_file, 'wb') as f:
            f.write(text_bytes)
        # bar.update(input_files.index(file) + 1)
    print('\nImport complete!')


def text_cleaning(embeddings=None):
    input_files = os.listdir(ORIGIN_DIRECTORY)
    # bar = progressbar.ProgressBar(len(input_files)).start()
    for file in input_files:
        with open(ORIGIN_DIRECTORY + file, 'rt') as f:
            text = f.read()
            # tokenization
            tokens = nltk.word_tokenize(text)

            # delete punctuation
            tokens = [word for word in tokens if word.isalpha()]

            # filter out stop words
            if not embeddings:
                stop_words = set(stopwords.words('russian'))
                tokens = [w for w in tokens if (w not in stop_words and w not in my_stopwords)]

            # convert to lowercase
            tokens = [w.lower() for w in tokens]

            # save file
            if embeddings:
                new_file = open(EMBEDDINGS_DIRECTORY + file + '_embeddings.txt', 'w')
            else:
                new_file = open(CLEAN_DIRECTORY + file + '_clean.txt', 'w')
            for token in tokens:
                new_file.write("%s " % token)
            # bar.update(input_files.index(file) + 1)
    print('\nCleaning complete!')
