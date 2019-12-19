from sklearn.model_selection import GridSearchCV
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os


class LdaApplication:
    def __init__(self, corpus, stopWords):
        self.corpus = corpus
        self.stopWords = stopWords
        self.tfVectorizer = TfidfVectorizer(stop_words=self.stopWords, min_df=0.002, max_df=0.98, max_features=256)
        self.tfVector = self.tfVectorizer.fit_transform(self.corpus)

    def _get_tf_vector(self):
        tfVector = self.tfVector
        return tfVector

    def train_model(self, n_components, learning_offset=10.0, learning_decay=0.7, max_doc_update_iter=100, n_jobs=-1):
        '''
        训练LDA模型
        :param n_components: LDA的主题参数
        :param learning_offset:
        :param learning_decay:
        :param max_doc_update_iter:
        :param n_jobs:
        :return: lda model
        '''
        ldaModel = LatentDirichletAllocation(n_components=n_components,
                                             learning_decay=learning_decay,
                                             learning_offset=learning_offset,
                                             max_doc_update_iter=max_doc_update_iter,
                                             n_jobs=n_jobs)
        ldaModel.fit(self.tfVector)
        print('The Log Likelihood Score:{}'.format(np.round(ldaModel.score(self._get_tf_vector()), 3)))
        print('The Perplexity:{}'.format(np.round(ldaModel.perplexity(self._get_tf_vector()), 3)))
        return ldaModel

    def grid_search_best_model(self, params):
        '''
        网格搜索训练最佳LDA依据极大似然值
        :param params: 网格搜索参数字典
        :return: best lda model
        '''
        ldaModel = LatentDirichletAllocation(n_jobs=-1)
        gridSearchModel = GridSearchCV(ldaModel, param_grid=params, cv=5)
        gridSearchModel.fit(self._get_tf_vector())
        bestLdaModel = gridSearchModel.best_estimator_
        pprint('The bet model params:{}'.format(gridSearchModel.best_params_))
        print('The best Log Likelihood Score:{}'.format(np.round(gridSearchModel.best_score_, 3)))
        print('The Perplexity:{}'.format(np.round(bestLdaModel.perplexity(self._get_tf_vector()), 3)))
        return bestLdaModel

    def result_display(self, model):
        pass

    def get_topic_distribution(self, model):
        '''
        展示主题分布
        :param model: 训练好的LDA模型
        :return: 词分布的df
        '''
        ldaOutput = model.transform(self._get_tf_vector())
        topicNames = ["Topic" + str(i) for i in range(model.n_components)]
        docNames = ["Doc" + str(i) for i in range(len(self.corpus))]
        dfDocumentTopic = pd.DataFrame(np.round(ldaOutput, 2), columns=topicNames, index=docNames)
        dominantTopic = np.argmax(dfDocumentTopic.values, axis=1)
        dfDocumentTopic['dominant_topic'] = dominantTopic
        dfTopicDistribution = dfDocumentTopic['dominant_topic'].value_counts().reset_index(name="Num Documents")
        dfTopicDistribution.columns = ['Topic Num', 'Num Documents']
        return dfTopicDistribution

    def get_top_n_words(self, model, n_words=10):
        '''
        展示每个主题下top_n的词
        :param model: 训练好的LDA 模型
        :param n_words: n的数量 默认10
        :return: 各个主题下top_n词的df
        '''
        keywords = np.array(self.tfVectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
        return df_topic_keywords

    def get_lda_vis(self, model, path='lda.html'):
        '''
        生成一个LDA结果可视化HTML
        :param path: Html 文件保存路径 默认 lda.html
        :param model: 需要可视化的LDA模型
        '''
        panel = pyLDAvis.sklearn.prepare(lda_model=model, vectorizer=self.tfVectorizer, dtm=self.tfVector, mds='tsne')
        # pyLDAvis.show(panel)
        pyLDAvis.save_html(panel, path)

    def predict(self, model, corpus):
        '''
        聚类新的文本并给出聚类结果
        :param model: 训练好的LDA模型
        :param corpus: 需要预测的语料
        :return: 语料对应聚类结果的矩阵
        '''
        tfVector = self.tfVectorizer.transform(corpus)
        ldaOutpu = model.transform(tfVector)
        results = np.argmax(ldaOutpu, axis=1)
        return results


if __name__ == '__main__':
    file = '../../../Data'
    corpusFile = 'LDA_demo_data.txt'
    stopWordsFile = 'stop_words.txt'
    with open(os.path.join(file, corpusFile), 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    with open(os.path.join(file, stopWordsFile), 'r', encoding='utf-8') as f:
        stopWords = f.read()
        stopWords = stopWords.splitlines()

    testCorpus = corpus[:10000]
    lda = LdaApplication(testCorpus, stopWords)
    model = lda.train_model(n_components=6, learning_decay=0.7)
    topicDis = lda.get_topic_distribution(model)
    print(topicDis)
    top_n_words = lda.get_top_n_words(model)
    print(top_n_words)
    lda.get_lda_vis(model)
