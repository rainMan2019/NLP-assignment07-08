import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time

# 用以统计函数执行时间
def time_func(f):
    def wrapper(*args):
        start = time()
        print('Starting processing....')
        result = f(*args)
        end = time()
        print('Process ended.....')
        duration = end - start
        print('----Processed in %ss----' % round(duration, 2))
        return result
    return wrapper

# 评价模型
def evaluate(clf, X, Y):
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    y_predicted = clf.predict(X)
    print('f1_score is: {}'.format(f1_score(Y, y_predicted)))
    print('accuracy is: {}'.format(accuracy_score(Y, y_predicted)))
    print('percision is: {}'.format(precision_score(Y, y_predicted)))
    print('recall is: {}'.format(recall_score(Y, y_predicted)))

def clean_cut(s):
    # 要将文本清洗、分词并以空格隔开，以便转为Tfidf向量表征
    # 清洗过程中发现有些文本还有\\n，也一并删除 
    return ' '.join(jieba.lcut(re.sub('[\r\n\u3000]', '', s).replace('\\n','')))


# 准备平衡的训练数据
@time_func
def prepare_inputs(df_train, feature):
    df_train[feature] = df_train[feature].fillna('').apply(clean_cut)
    df_train['is_xinhua'] = np.where(df_train['source'].str.contains('新华'), 1, 0)
    x_inputs = df_train[feature]
    y_inputs = df_train['is_xinhua']
    return x_inputs, y_inputs


# 避免训练数据不平衡，这里选取等量的两部分数据进行训练
def sample_train_data(df):
    df = df.fillna('')
    df_notxinhua = df[df.source != '新华社'][df.content != '']
    df_xinhua = df[df.source == '新华社']
    df_train = df_xinhua.sample(n=len(df_notxinhua))
    df_train = df_train.append(df_notxinhua)
    return df_train

# 训练KNN模型
def knn_train(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    # 调参
    knn_clf = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform', algorithm = 'auto')
    # 训练模型
    knn_clf.fit(x_train, y_train)
    return y_predicted

# 找出潜在的抄袭作品，具体条件为预测来源为新华社，但实际又不是的
def find_potential_copies(y_test, y_predicted):
    potential_copy_num = [no for no, is_xinhua in enumerate(y_test) if is_xinhua == 0 and y_predicted[no] == 1]
    potential_copy_index = [y_test.index[no] for no in potential_copies ]
    potential_copy = {df_train.source[p]: df_train.content[p] for p in potential_copy_index}
    return potential_copies
    

if __name__ == "__main__":
    df = pd.read_csv('news.csv', encoding='gb18030-2000')
    df_train = sample_train_data(df)
    x_inputs, y_inputs = prepare_inputs(df_train, 'content')
    # 必须要设置max_feature，否则训练好后精确度很低
    vectorizer = TfidfVectorizer(max_features=600)
    # 要将文字转为tf-idf向量表示
    X = vectorizer.fit_transform(x_inputs.values)
    Y = y_inputs
    x_train, x_test, y_train, y_test = train_test_split(
        X , Y, train_size = 0.9, test_size=0.1
    )
	knn_clf = knn_train(x_train, y_train)
    knn_y_predicted = knn_clf.predict(x_test)
	evaluate(knn_clf, X, Y)
    potential_copies = find_potential_copies(y_test, knn_y_predicted)