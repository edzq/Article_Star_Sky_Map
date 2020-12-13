import pandas as pd
import csv
import jieba
import gensim as gensim
import re
import math
import numpy as np

from tsne_add_1 import *
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence


from sklearn.manifold import TSNE
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import copy

def map_linear(x, floor, ceil):
    darray_x = np.array(x)
    floor_x = np.min(x)
    ceil_x = np.max(x)
    darray_x = (darray_x - floor_x)/(ceil_x - floor_x)*(ceil - floor)+floor
    return darray_x

def load_stopword(filename):
    '''
    加载停止词
    :return:
    '''
    with open(filename,'r',encoding='utf-8') as f:
        stopword = [line.strip() for line in f]
    return stopword

def extract(contents, stopword):
    '''
    用正则表达式去除无用内容以及停止词，保留有用的内容
    :param contents: list
    :param stopword: list
    :return:
    '''
    extracted_contents = []
    for content in contents:
        extracted_words = []
        content = str(content).strip()
        re_h = re.compile('</?\w+[^>]*>')
        re_nbsp = re.compile('&nbsp;')
        content = re_h.sub('', content)
        content = re_nbsp.sub('', content)
        words = jieba.lcut(content)
        for w in words:
            w = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", w)
            if len(w)>0 and w not in stopword:
                extracted_words.append(w)
        #if len(extracted_words) > 0:
        extracted_contents.append(extracted_words)
    return extracted_contents

def get_Content(str_list):
    end_sign = []
    content = []
    content_length = []
    start_idx = 0
    end_idx = 0 
    period_idx = 0
    for i in range(len(str_list)-1):
        st = str_list[i]
        st_next = str_list[i+1]
        if st == '>':
            start_idx = i
        if st == '<' and st_next == '/':
            end_idx = i
        if start_idx > end_idx and end_idx>0:
            if start_idx > end_idx + 1:
                content.append(str_list[period_idx+1:end_idx])
            else:
                content.append([])
            content_length.append(len(content[-1]))
            end_sign.append(str_list[end_idx:start_idx+1])
        period_idx = start_idx
    
    new_content = []
    for i in range(len(content)):
        if content_length[i]>0:
            if end_sign[i] not in['</script>','</style>']:
                new_content.append(content[i])
    return new_content

def Cosine(vec_1,vec_2):
    vec_1,vec_2 = np.array(vec_1),np.array(vec_2)
    resemble = vec_1.dot(vec_2)/(math.sqrt(np.sum(vec_1*vec_1)*np.sum(vec_2*vec_2)))
    
    return resemble

def movealittle(size, axis, iteration=1):
    """
    size: 每个小球的半径大小
    axis: 每个小球的初始坐标
    """
    ###iterate
    for time in range(iteration):
        X = axis[:,0]
        Y = axis[:,1]
        x = list(X)
        y = list(Y)
        disX = []
        disY = []
        sizedis = []
        for i in range(len(axis)):
            disX.append(x)
            disY.append(y)
            sizedis.append(size)
        disX = np.array(disX)
        disY = np.array(disY)
        disX = disX - disX.T
        disY = disY - disY.T
        sizedis = np.array(sizedis)
        sizedis = sizedis.T + sizedis
        #这一句话很重要～
        dis = np.maximum(np.sqrt(disX**2 + disY**2),0.01)

        for i in range(len(dis)):
            for j in range(len(dis)):
                if i == j :
                    dis[i,j] = 1

        delta = sizedis - dis
        delta1 = [[max(0,delta[r,c]) for c in range(len(delta[r]))] for r in range(len(delta))]
        deltax = delta1*disX/dis*0.1
        deltay = delta1*disY/dis*0.1

        for i in range(len(dis)):
            for j in range(len(dis)):
                if i == j :
                    deltax[i,j] = 0
                    deltay[i,j] = 0
        X = X + deltax.sum(axis = 0)
        Y = Y + deltay.sum(axis = 0)
        X = np.array([X,Y])
        axis = X.T
        print(X.shape)
    return axis

def generate_core_axis(df, cate_num, category_name):
    '''
    获取星空图个个类别的中心位置，并写入coursecore_now.txt当中,category,x,y
    :param df: 课程的类别以及坐标
    :return:
    '''
    #category_name = ['规模视角下的宏观与微观','人工智能与社会','技术控','计算社会科学','自然语言处理','聊天机器人','实验基础Mathematics','数据与社会','系统科学','数学物理与宇宙','Julia']
    #cate_num = 11
    X_core = np.zeros((cate_num,2))
    df_courseaxis = df
    for cate in range(cate_num):
        df_courseaxis_sub = df_courseaxis[df_courseaxis['category']==cate]
        X_core[cate,0] = df_courseaxis_sub['x'].mean()
        X_core[cate,1] = df_courseaxis_sub['y'].mean()
    df_article_core = pd.DataFrame({'category_information':category_name,'x':X_core[:,0],'y':X_core[:,1],'category':list(range(cate_num))})
    df_courseaxis.to_csv('articlecore.csv')
    with open('Insert_articlecore.txt','w',encoding='utf-8') as f:
        for i in range(len(X_core)):
            c_id = i
            #view = str(df_core_name[i,2]*20)
            name = str(category_name[i])
            #category = str(df_core_name[i,3])
            x = str(X_core[i,0])
            y = str(X_core[i,1])
            sql = "(" + str(i+1) + ",'" + str(i) + "','" + name + "','" + x + "','" + y + "')"
            f.write(sql+','+'\n')

 

def load_word2vec():
    word_vectors = KeyedVectors.load_word2vec_format('veclib.bin', binary=True, unicode_errors='ignore')
    rawWordVec_2 = []
    word2ind_2 = {}
    for i, w in enumerate(word_vectors.vocab):
        rawWordVec_2.append(word_vectors[w])
        word2ind_2[w] = i
    rawWordVec_2 = np.array(rawWordVec_2)
    return rawWordVec_2, word2ind_2

def sort_the_order(Y_original,idx_old,idx_new,A):
    '''
    根据以前的坐标/类别获得新的课程的坐标/类别
    :param Y_original: 上一版课程的坐标/类别
    :param idx_old: 上一版课程的id 序列
    :param idx_new: 这一版课程的id序列
    :param A: 相似矩阵
    :return:新的课程的坐标/类别，array
    '''
    Y_original = np.array(Y_original)
    Y_present = []
    A_sub = np.zeros((len(A),len(A)))
    A_sub[pd.Series(idx_new).isin(idx_old)] = 1
    A_sub = A*A_sub.T
    for idx in idx_new:
        if idx in idx_old:
            oidx = idx_old.index(idx)
            Y_present.append(Y_original[oidx])
        else:
            resemble_sort = np.argsort(A_sub[idx_new.index(idx)])
            cid_most_similar = idx_new[resemble_sort[-1]]
            if cid_most_similar in idx_old:
                oidx = idx_old.index(cid_most_similar)
                Y_present.append(Y_original[oidx])
            else:
                print(idx,'brand new courses')
                Y_present.append(np.zeros(Y_original.shape[1]))
    return np.array(Y_present)



def main(): 
    jieba.load_userdict('lexicon.txt')
    jieba.load_userdict('lexicon_by_manual.txt')
    jieba.load_userdict('keyword_1_py.txt')
    jieba.load_userdict('topic.txt')

    lexicon = load_stopword('lexicon.txt')
    lexicon_M = load_stopword('lexicon_by_manual.txt')
    keyword_py = load_stopword('keyword_1_py.txt')
    tag = load_stopword('topic.txt')
    keyword_py.extend(tag)

    stopword = load_stopword('stopword.txt')
    extract_out = ['showDate','877391004','article','gpac','ori','https','◆','QQ','videoPlayerIconSpan','赞赏','投稿','原文','天前','一扫','©','tmp','周前','昨天','取消','desc']
    stopword.extend(extract_out) 

    df_detail = pd.read_csv('article_details.csv')
    artid_list = list(df_detail['ID'])

    count = 0
    sentence_all = []
    content = []
    original_content_html = list(df_detail['post_content'])
    for i in range(len(df_detail)):
        content.append(get_Content(original_content_html[i]))
        extract_content = extract(content[-1],stopword)
        sentence = []
        for each in extract_content:
            sentence.extend(each)
        sentence_all.append(sentence)
        print(i)
    """
    vocab = []
    for sentence in sentence_all:
        for w in sentence:
            vocab.append(w.lower())
    vocab = list(set(vocab))
    rawWordVec_2, word2ind_2 = load_word2vec()
    """
    keyword_py_lo = [w.lower() for w in keyword_py]
    """
    vocab_invec = []
    keyword_notinvec = []
    for w in vocab:
        if w.lower() in word2ind_2:
            vocab_invec.append(w.lower())
        else:
            if w.lower() in keyword_py_lo:
                keyword_notinvec.append(w.lower())

    art_vec_bow = []
    for sentence in sentence_all:
        seq = np.zeros(200)#wordvec's dimension
        seq_keywordscore = np.zeros(len(keyword_notinvec))
        for w in sentence:
            w_lo = w.lower()
            if w_lo in word2ind_2:
                if w_lo in keyword_py_lo:
                    seq += rawWordVec_2[word2ind_2[w_lo]]*20
                else:
                    seq += rawWordVec_2[word2ind_2[w_lo]]
            if w_lo in keyword_notinvec:
                seq_keywordscore[keyword_notinvec.index(w_lo)] += 50
        art_vec_bow.append(np.hstack((seq,seq_keywordscore))) 
        print(len(art_vec_bow))
    """
    art_vec_bow = []
    count = 0
    for sentence in sentence_all:
        series = np.zeros(len(keyword_py_lo))
        for w in sentence:
            if w.lower() in keyword_py_lo:
                series[keyword_py_lo.index(w.lower())] += 1
        art_vec_bow.append(series)
        series[-1] = count
        count += 1
    art_vec_bow.append(series)  
    art_vec_bow = np.array(art_vec_bow)

    A = np.zeros((len(sentence_all),len(sentence_all)))
    for i in range(len(sentence_all)):
        for j in range(i+1, len(sentence_all)):
            A[i,j] = Cosine(art_vec_bow[i],art_vec_bow[j]) 
            A[j,i] = Cosine(art_vec_bow[i],art_vec_bow[j])

    df_article_original = pd.read_csv('articleaxis_now.csv')###以前的课程信息
    artid_now = list(df_article_original['artid'])

    Y_original = np.array(df_article_original[['x','y']])
    #course_cate_now = np.array(df_courseaxis_original['category'])
    Y_original = sort_the_order(Y_original, artid_now, artid_list, A)

    X_reduced = tsne(A,Y_original,n=2,neighbors=30,I=0.00001,max_iter=200)
    df_core = pd.read_csv('articlecore.csv')###加入类别标号与名称
    df_core = df_core.sort_values(by='category')
    category_name = list(df_core['category_information'])
    article_core = np.array(df_core[['x','y']]) 
    
    model_k = KMeans(n_clusters = 20, init=article_core, n_jobs = 4, max_iter = 50) #分为k类, 并发数4
    model_k.fit(X_reduced)
    article_cate = model_k.labels_

    ## adjust the view to the range of [15,54] and also obey the power law
    a = 0.4
    # 采样数量
    samples = len(df_detail)
    s = np.random.power(a, samples)
    s = map_linear(s,15,54)
    #s = sorted(s,reverse = True)
    #s = np.maximum(0.1,s)
    view = np.array(df_detail['meta_value'])
    view_0 = [np.sort(s)[list(np.argsort(view)).index(i)] for i in range(len(np.argsort(view)))]
    """
    view_0 = view_0/np.exp(18)
    view_1 = adjust_difference(adjust_cid,cid_list,view_0)
    print("view value distribution description:")
    """
    print(pd.Series(view_0).describe())

    X_reduced0 = copy.copy(X_reduced)
    view_move = np.array(view_0)/12
    X_reduced = movealittle(view_move,X_reduced0,10)


    df_courseaxis_now = pd.DataFrame({'view':view_0,'artid':list(df_detail['ID']),'title':list(df_detail['post_title']),'category':article_cate,'x':X_reduced[:,0],'y':X_reduced[:,1]})
    df_courseaxis_now = df_courseaxis_now[['artid','view','title','category','x','y']]
    df_courseaxis_now.to_csv('articleaxis_now.csv',encoding = 'utf-8')
    df_table = np.array(df_courseaxis_now)
    with open('articleaxis_Insert.txt','w',encoding='utf-8') as f:
        for i in range(len(df_table)):
            c_id = str(df_table[i,0])
            view = str(view_0[i]*15)
            name = df_table[i,2]
            category = str(df_table[i,3])
            x = str(df_table[i,4])
            y = str(df_table[i,5])
            sql = "(" + str(i+1) + "," + c_id + "," + view + ",'" + name + "','" + category + "','" + x + "','" + y + "')"
            f.write(sql+','+'\n')
    print('new articleAxis has been written into file')

    df_courseaxis_now.to_excel('articleaxis_now.xlsx')



    #category_name = list(range(20))
    generate_core_axis(df_courseaxis_now,20,category_name)
 
if __name__ == '__main__':
    main()


