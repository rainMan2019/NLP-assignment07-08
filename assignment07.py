import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time

# ����ͳ�ƺ���ִ��ʱ��
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

# ����ģ��
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
    # Ҫ���ı���ϴ���ִʲ��Կո�������Ա�תΪTfidf��������
    # ��ϴ�����з�����Щ�ı�����\\n��Ҳһ��ɾ�� 
    return ' '.join(jieba.lcut(re.sub('[\r\n\u3000]', '', s).replace('\\n','')))


# ׼��ƽ���ѵ������
@time_func
def prepare_inputs(df_train, feature):
    df_train[feature] = df_train[feature].fillna('').apply(clean_cut)
    df_train['is_xinhua'] = np.where(df_train['source'].str.contains('�»�'), 1, 0)
    x_inputs = df_train[feature]
    y_inputs = df_train['is_xinhua']
    return x_inputs, y_inputs


# ����ѵ�����ݲ�ƽ�⣬����ѡȡ���������������ݽ���ѵ��
def sample_train_data(df):
    df = df.fillna('')
    df_notxinhua = df[df.source != '�»���'][df.content != '']
    df_xinhua = df[df.source == '�»���']
    df_train = df_xinhua.sample(n=len(df_notxinhua))
    df_train = df_train.append(df_notxinhua)
    return df_train

# ѵ��KNNģ��
def knn_train(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    # ����
    knn_clf = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform', algorithm = 'auto')
    # ѵ��ģ��
    knn_clf.fit(x_train, y_train)
    return y_predicted

# �ҳ�Ǳ�ڵĳ�Ϯ��Ʒ����������ΪԤ����ԴΪ�»��磬��ʵ���ֲ��ǵ�
def find_potential_copies(y_test, y_predicted):
    potential_copy_num = [no for no, is_xinhua in enumerate(y_test) if is_xinhua == 0 and y_predicted[no] == 1]
    potential_copy_index = [y_test.index[no] for no in potential_copies ]
    potential_copy = {df_train.source[p]: df_train.content[p] for p in potential_copy_index}
    return potential_copies
    

if __name__ == "__main__":
    df = pd.read_csv('news.csv', encoding='gb18030-2000')
    df_train = sample_train_data(df)
    x_inputs, y_inputs = prepare_inputs(df_train, 'content')
    # ����Ҫ����max_feature������ѵ���ú�ȷ�Ⱥܵ�
    vectorizer = TfidfVectorizer(max_features=600)
    # Ҫ������תΪtf-idf������ʾ
    X = vectorizer.fit_transform(x_inputs.values)
    Y = y_inputs
    x_train, x_test, y_train, y_test = train_test_split(
        X , Y, train_size = 0.9, test_size=0.1
    )
	knn_clf = knn_train(x_train, y_train)
    knn_y_predicted = knn_clf.predict(x_test)
	evaluate(knn_clf, X, Y)
    potential_copies = find_potential_copies(y_test, knn_y_predicted)