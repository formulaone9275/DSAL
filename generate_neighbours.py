#import sent2vec
from build_tfrecord_data_new import read_sentences,sentence_sequence,load_pickle
import pickle
from numpy import dot
from math import sqrt
import numpy as np
from cython.parallel import prange
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
def cosine_similarity(x,y):
    if sqrt(dot(x,x))==0 or sqrt(dot(y,y))==0:
        return 0
    return dot(x,y)/(sqrt(dot(x,x))*sqrt(dot(y,y)))

def get_sent_emb(file_name):
    model = sent2vec.Sent2vecModel()
    model.load_model('/home/psu/Documents/sent2vec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')

    #file_name='ppi'
    sent,dep_sents,label=read_sentences('./data/ds/'+file_name+'.txt')
    sent_mx=sentence_sequence(sent)


    emb_sent = model.embed_sentences(sent_mx)
    emb_dep=model.embed_sentences(dep_sents)
    with open('./data/ds/sent_emb_'+file_name+'.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(emb_sent, f, pickle.HIGHEST_PROTOCOL)
    with open('./data/ds/dep_emb_'+file_name+'.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(emb_dep, f, pickle.HIGHEST_PROTOCOL)

def get_similarity_score(sent_pickle,dep_pickle,alpha):
    #load the pickle files
    sent_mx=load_pickle('',sent_pickle)
    dep_mx=load_pickle('',dep_pickle)
    sent_num=len(sent_mx)
    score_dic={}

    #calculate the similarity
    for ii in tqdm(prange(sent_num)):
        # ii means the ii-th sentence
        similarity_arr=np.zeros(sent_num)
        similarity_arr=list(similarity_arr)
        for jj in range(sent_num):
            if jj==ii:
                similarity_arr[jj]=1
            else:
                similarity_arr[jj]=alpha*cosine_similarity(sent_mx[ii],sent_mx[jj])+(1-alpha)*cosine_similarity(dep_mx[ii],dep_mx[jj])

        top_k_nearest=np.argsort(similarity_arr)[-100:]
        top_k_score=[similarity_arr[i] for i in top_k_nearest]
        score_dic[str(ii)]=(top_k_nearest,top_k_score)
        #print(score_dic[str(ii)])
    with open('./data/ds/score_dic.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(score_dic, f, pickle.HIGHEST_PROTOCOL)

    return score_dic


#define a new function for the for loop
def cal_sim_arr(index2,index1,score_list,alpha_inside,sent_pickle,dep_pickle):

    #load the pickle files
    sent_mx=load_pickle('',sent_pickle)
    dep_mx=load_pickle('',dep_pickle)

    overall_score=alpha_inside*cosine_similarity(sent_mx[index1],sent_mx[index2])+(1-alpha_inside)*cosine_similarity(dep_mx[index1],dep_mx[index2])
    score_list[index2]=overall_score
    return score_list

def get_similarity_score_mp(sent_pickle,dep_pickle,alpha):



    sent_mx_=load_pickle('',sent_pickle)
    sent_num=len(sent_mx_)
    #build the similarity matrix first and then generate the score_dic

    score_dic={}
    #calculate the similarity
    for ii in tqdm(range(sent_num)):
        # ii means the ii-th sentence
        similarity_arr=np.zeros(sent_num)
        similarity_arr=list(similarity_arr)
        with Pool(processes=10) as pool:
            similarity_arr = pool.map(partial(cal_sim_arr, index1=ii,score_list=similarity_arr,alpha_inside=alpha,sent_pickle=sent_pickle,dep_pickle=dep_pickle), [ind for ind in range(sent_num)])
            top_k_nearest=np.argsort(similarity_arr)[-100:]
            top_k_score=[similarity_arr[i] for i in top_k_nearest]
            print('ii',ii)
        score_dic[str(ii)]=(top_k_nearest,top_k_score)
    with open('./data/ds/similarity_dic.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(score_dic, f, pickle.HIGHEST_PROTOCOL)

    return score_dic

def update_label(score_dic,labels,k):
    #count the changing of labels
    changed_label=0
    max_iter=10
    while max_iter>0:
        #find the k nearest neightbours
        for ni in score_dic.keys():
            top_k_nearest=np.argsort(score_dic[ni])[-1*k:]
            #get the label of top k
            neighbour_label_set=[]
            for ti in top_k_nearest:
                neighbour_label_set.append(labels[ti])
            postive_label_num=neighbour_label_set.count([0, 1])
            negative_label_num=neighbour_label_set.count([1, 0])

            if postive_label_num>negative_label_num:
                decided_label=[0,1]
            else:
                decided_label=[1,0]
            #change the original label or not
            if labels[int(ni)]!= decided_label:
                labels[int(ni)]=decided_label
                changed_label+=1
        if changed_label==0:
            break

    return labels

if __name__ == '__main__':

    file_name='ppi_filtered'
    #get_sent_emb(file_name)
    sent_pickle='./data/ds/sent_emb_'+file_name+'.pickle'
    dep_pickle='./data/ds/dep_emb_'+file_name+'.pickle'
    alpha=0.5
    s_dic=get_similarity_score(sent_pickle,dep_pickle,alpha)
    print(s_dic)