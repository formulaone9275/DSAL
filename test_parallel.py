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



def get_similarity_score_one(sent_indx):

    file_name='ppi_filtered'
    #get_sent_emb(file_name)
    sent_pickle='./data/ds/sent_emb_'+file_name+'.pickle'
    dep_pickle='./data/ds/dep_emb_'+file_name+'.pickle'
    alpha=0.5

    #load the pickle files
    sent_mx=load_pickle('',sent_pickle)
    dep_mx=load_pickle('',dep_pickle)
    sent_num=len(sent_mx)


    similarity_arr=np.zeros(sent_num)
    similarity_arr=list(similarity_arr)
    #calculate the similarity
    for ii in range(sent_num):


        if sent_indx==ii:
            similarity_arr[ii]=1
        else:
            similarity_arr[ii]=alpha*cosine_similarity(sent_mx[ii],sent_mx[sent_indx])+(1-alpha)*cosine_similarity(dep_mx[ii],dep_mx[sent_indx])

    top_k_nearest=np.argsort(similarity_arr)[-100:]
    top_k_score=[similarity_arr[i] for i in top_k_nearest]
    sent_top=(top_k_nearest,top_k_score)
    with open('./data/ds/sent_sim/score_top'+str(sent_indx)+'.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(sent_top, f, pickle.HIGHEST_PROTOCOL)

    print(sent_indx)
    return sent_top


def get_similarity_score():

    with Pool(processes=10) as pool:
        pool.map(get_similarity_score_one, [i for i in range(128085)])

    '''
    for si in tqdm(prange(128085)):
        sent_top=get_similarity_score_one(si)
    '''
    return 0

if __name__ == '__main__':

    s_dic=get_similarity_score()
    print(s_dic)