from __future__ import print_function
import codecs
import tensorflow as tf
import numpy as np

import random
import sys
import pickle,os




def read_sentences(filename):
    sentences, dep_sents,  labels = [], [], []
    count = 0

    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            if line.startswith(('NONE', 'R_PPI')):


                count += 1

                line = line.strip()
                info, tagged,dep_path = line.split('\t')
                eles = info.split(' ')
                label = eles[0]
                sent = []
                head = []

                tokens = [e for e in eles if e.startswith('token:')]
                for token in tokens:
                    token = token[6:]
                    # Sometimes weird word contains the vertical bar.
                    word, pos, ent_type, to_e1, to_e2, dep_label, dep_head,e1_head,e2_head = token.rsplit('|',8)
                    to_e1 = int(to_e1)
                    to_e2 = int(to_e2)

                    sent.append((word, pos, ent_type, to_e1, to_e2, dep_label))



                sentences.append(sent)
                dep_sents.append(dep_path)
                labels.append([0, 1] if label.startswith('R_PPI') else [1, 0])
            elif line.startswith(('Positive', 'Negative')):


                count += 1

                line = line.strip()
                info, tagged,dep_path = line.split('\t')
                eles = info.split(' ')
                label = eles[0]
                sent = []
                head = []

                tokens = [e for e in eles if e.startswith('token:')]
                for token in tokens:
                    token = token[6:]
                    # Sometimes weird word contains the vertical bar.
                    word, pos, ent_type, to_e1, to_e2, dep_label, dep_head = token.rsplit('|')
                    to_e1 = int(to_e1)
                    to_e2 = int(to_e2)

                    sent.append((word, pos, ent_type, to_e1, to_e2, dep_label))



                sentences.append(sent)
                dep_sents.append(dep_path)
                labels.append([0, 1] if label.startswith('Positive') else [1, 0])
            else:
                continue



    return sentences,dep_sents, labels


def sentence_sequence(sentences, mask_p1p2=True,mask_other=None):
    sequence_matrix = []

    token_count = 1
    missing_embed = 0
    missing_mapped = 0
    for sent in sentences:
        sentence_seq=''

        previous_entity_type='new'
        for (word, pos, ent_type, to_e1, to_e2, dep_label) in sent:

            token_count += 1

            if mask_p1p2 and (ent_type in ['P1','P1P2','PROT1','PROT12']):
                if previous_entity_type==ent_type:
                    continue
                else:
                    word = 'PROTEIN1'
            if mask_p1p2 and (ent_type in [ 'P2','PROT2']):
                if previous_entity_type==ent_type:
                    continue
                else:
                    word = 'PROTEIN2'
            if mask_other and (ent_type == 'P' or ent_type == 'PROT'):
                if previous_entity_type==ent_type:
                    continue
                else:
                    word = 'PROTEIN'
            previous_entity_type=ent_type
            if word in [',',')','.']:
                sentence_seq+=word
            else:
                sentence_seq+=(' '+word)
            sentence_seq=sentence_seq.replace('( ','(')
        sequence_matrix.append(sentence_seq)

    return sequence_matrix

def load_pickle(pickle_file_path,file_name):


    with open(pickle_file_path+file_name, 'rb') as f:
        pickle_var = pickle.load(f,encoding="latin1")

    return pickle_var

def random_and_divide_file_v0(filename,output_folder,number):
    file_dict={}
    count_line=0

    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            file_dict[count_line] = line.strip()

            count_line+=1
    index_list=list(range(len(file_dict)))
    random.shuffle(index_list)


    for i in range(number):
        file_object=codecs.open(output_folder+"fold"+str(i+1)+".txt",'w+',encoding='utf8')
        for j in range(int(len(file_dict)/number)*i,int(len(file_dict)/number)*(i+1)):

            file_object.write(file_dict[index_list[j]])
            file_object.write('\n')
        file_object.close()


if __name__ == '__main__':
    #random_and_divide_file_v0('./data/aimed.txt','./data/aimed_instance/',10)

    sent,dep_sents,label=read_sentences('./data/aimed_instance/fold1.txt')
    sent_mx=sentence_sequence(sent)
    print(sent_mx)
    print(dep_sents)
    print(label)













