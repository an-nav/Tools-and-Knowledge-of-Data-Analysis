import numpy as np
import pandas as pd
import re
import codecs
import jieba
from utils import choose


class LDA:
    def __init__(self, num_topics, corpus, stop_words, alpha=None, eta=0.1, max_iteration=10):
        """
        data preprocess that will generate a 2D document list contain splited-cleaned corpus,
                                             a word to id dictionary,
                                             a id to word dictionary
        :param num_topics: int; the number of topics
        :param corpus: 2D list; the inner list is the list of word from each document and the outter list contains inner lists
        :param stop_words: list; stop words list
        :param alpha: float; the priori of  doc-topic dirichlet distribution
        :param eta:float; the priori of  topic-word dirichlet distribution
        :param max_iteration: int; the max iteration of gibbs sampling
        """
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = float(50 / num_topics)
        self.eta = eta
        self.K = num_topics
        self._corpus = corpus
        self.max_iteration = max_iteration
        self.word2id = {}
        self.id2word = {}
        self.document = []
        index = 0
        for doc in corpus:
            word_count = {}
            temp_doc = []
            for word in doc:
                word = word.lower()
                if word not in stop_words and len(word) > 1 and not re.search(r'[0-9]', word):
                    temp_doc.append(word)
                    if word not in self.word2id.keys():
                        self.word2id[word] = index
                        self.id2word[index] = word
                        index += 1
                    if word in word_count.keys():
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
            self.document.append(temp_doc)
        # number of docs
        self.M = len(self._corpus)
        # number of words
        self.N = len(self.word2id)
        self.doc_topic_matrix = np.zeros([self.M, self.K], dtype=np.int8)
        self.topic_word_matrix = np.zeros([self.K, self.N], dtype=np.int8)
        self.topic_matrix = np.zeros(self.K, dtype=np.int8)
        self.current_word_topic_matrix = []

    def _initialize(self):
        """
        initialize the doc_topic_matrix, topic_word_matrix and topic_matrix with a random topic number
        """
        for doc_index, doc in enumerate(self.document):
            temp_word_topic_matrix = []
            for word in doc:
                if word in self.word2id.keys():
                    start_topic_index = np.random.randint(0, self.K)
                    temp_word_topic_matrix.append(start_topic_index)
                    self.doc_topic_matrix[doc_index, start_topic_index] += 1
                    self.topic_word_matrix[start_topic_index, self.word2id[word]] += 1
                    self.topic_matrix[start_topic_index] += 1
            self.current_word_topic_matrix.append(temp_word_topic_matrix)

    def train(self):
        """
        Gibbs sampling
        """
        self._initialize()
        for i in range(self.max_iteration):
            print('iteration:{}'.format(i+1))
            for doc_index, doc in enumerate(self.document):
                for word_index, word in enumerate(doc):
                    current_topic_index = self.current_word_topic_matrix[doc_index][word_index]
                    # exclude the counts related to current topic
                    self.doc_topic_matrix[doc_index, current_topic_index] -= 1
                    self.topic_word_matrix[current_topic_index, word_index] -= 1
                    self.topic_matrix[current_topic_index] -= 1
                    # (n_{d,-i}^k+a_k)*(n_{k,-i}^t+b_t)/sum_t(n_{k,-i}^t+b_t)
                    topic_distribution = (self.doc_topic_matrix[doc_index] + self.alpha) *\
                                         (self.topic_word_matrix[:, word_index] + self.eta) /\
                                         (self.topic_matrix[current_topic_index] + self.N * self.eta)
                    new_topic_index = choose(range(self.K), topic_distribution)
                    self.current_word_topic_matrix[doc_index][word_index] = new_topic_index
                    # add the counts related to new topic
                    self.doc_topic_matrix[doc_index, new_topic_index] += 1
                    self.topic_word_matrix[new_topic_index, word_index] += 1
                    self.topic_matrix[new_topic_index] += 1

    def transform(self):
        """
        get the topic distribution of trained data
        :return: numpy array contain the topic distribution of each document
        """
        result = []
        for item in self.doc_topic_matrix:
            result.append(item / np.sum(item))
        result = np.array(result)
        return result

    def predict(self, data, max_iteration=20, tol=1e-16):
        """
        predict single the topic distribution of single document
        :param data: array_like shape(1, n_features); predicted data
        :param max_iteration: int; max iterations
        :param tol: float; the number to control the stopping of iteration
        :return: array_like shape(1, n_topic); the topic distribution of predicted data

        question remained: what's the mathematics function of prediction process
        """
        doc_topic_matrix = np.zeros([len(data), self.K], dtype=np.float)
        word_index_list = []
        for word in data:
            word_index_list.append(self.word2id[word])
        for i in range(max_iteration + 1):
            doc_topic_matrix_new = self.topic_word_matrix[:, word_index_list].T
            doc_topic_matrix_new = doc_topic_matrix_new.astype(np.float)
            doc_topic_matrix_new *= (doc_topic_matrix_new.sum(axis=0) - doc_topic_matrix + self.alpha)
            doc_topic_matrix_new /= doc_topic_matrix_new.sum(axis=1)[:, np.newaxis]
            delta_naive = np.abs(doc_topic_matrix_new - doc_topic_matrix).sum()
            doc_topic_matrix = doc_topic_matrix_new
            if delta_naive < tol:
                break
        theta_doc = doc_topic_matrix.sum(axis=0) / doc_topic_matrix.sum()
        return theta_doc

    def getTopNWords(self, n=5):
        """
        get top n word of each topic
        :param n: the number of top words
        :return: a dataframe which the index is the topic_num, the column is the word_num
        """
        word_id = []
        for i in range(self.topic_word_matrix.shape[0]):
            word_id.append(self.topic_word_matrix[i].argsort()[:n])
        top_word_df = pd.DataFrame(index=['topic{}'.format(x) for x in range(self.K)],
                                   columns=['word{}'.format(x) for x in range(n)])
        for i in range(len(word_id)):
            for j in range(n):
                top_word_df.loc['topic{}'.format(i), 'word{}'.format(j)] = self.id2word[word_id[i][j]]
        return top_word_df


def test():
    data = codecs.open('./data/dataset.txt', encoding='utf-8')
    stop_words = codecs.open('./data/stopwords.dic', encoding='utf-8')

    data = [x.strip() for x in data]
    stop_words = [x.strip() for x in stop_words]
    docs = [jieba.cut(x) for x in data]

    lda = LDA(num_topics=3, corpus=docs, stop_words=stop_words, alpha=0.1, eta=0.1, max_iteration=10)
    lda.train()
    train_result = lda.transform()
    print('train result:\n', train_result)
    # use the last document to test predict()
    test_data = np.array(lda.document[-1])
    test_result = lda.predict(test_data)
    print('test result:\n', test_result)


if __name__ == '__main__':
    test()


