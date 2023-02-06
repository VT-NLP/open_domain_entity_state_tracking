import random
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F
from enum import Enum
import torch.nn as nn
import json
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")


"""----------------data progress-------------"""
def get_location_entity_relation(sentence):
    sentence=sentence.replace(".", " ").replace("_", " ")
    sentence=sentence.split("#")[:-1]
    #sentence: ['denver colorado', 'location location time zones', 'x']
    #list_positions #[[0, 1], [2, 3, 4, 5, 6], [7]]
    i=-1
    list_positions=[]
    for s in sentence:
        s=s.split(" ")
        position=[]
        for word in s:
            i=i+1
            position.append(i)
        list_positions.append(position)
    #print(list_positions) #[[0, 1], [2, 3, 4, 5, 6], [7]]
    return list_positions






    # return 0


def load_data(fpath):
    question = []
    poss=[]
    negs=[]
    pos_positions=[]
    neg_positions=[]
    with open(fpath, encoding="utf-8") as f2:
        S_length = []
        R = 0
        al = 0
        for line in f2:
            al = al + 1
            input_json = json.loads(line)
            DIC = {}
            q= input_json["question"]
            pos=input_json["ground_attri"].split(",")
            candidate=input_json["entities_within_hop2"].split(",")

            neg=[]
            for cand in candidate:
                if cand not in pos:
                    neg.append(cand)


            if len(neg) > len(pos) or len(neg)==len(pos):
                neg = random.sample(neg, len(pos))
            else:
                neg =[]
                if len(pos)!=0 and len(neg)!=0:
                    for i in range(len(pos)):
                        neg.append(random.sample(neg,1))
                else:
                    neg=pos


            neg_length = len(neg)

            for i in range(neg_length):
                question.append(q)
                poss.append(pos[i])
                negs.append(neg[i])

    return question,poss,negs







def load_test_data1(fpath):
    T=[]

    with open(fpath, encoding="utf-8") as f2:
        S_length = []
        R = 0
        al = 0
        for line in f2:
            triple1 = []
            # al = al + 1
            # print(al)
            input_json = json.loads(line)
            DIC = {}
            id=input_json["id"]
            answers=input_json["answer"]
            q= input_json["question"]
            pos=input_json["ground_attri"].split(",")
            candidate=input_json["entities_within_hop2"].split(",")
            triple1.append(q)
            triple1.append(pos)
            triple1.append(candidate)
            triple1.append(id)
            triple1.append(answers)
            T.append(triple1)

    return T






"""----------------Loss function-------------"""

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)



class TripletLossC(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    """
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 5):
        super(TripletLossC, self).__init__()


        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def forward(self, rep_anchor, rep_pos, rep_neg):

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)

        return losses.mean()


"""----------------data to dataloader in torch-------------"""

class DatasetChenQA(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        return data_point


def _collate_fn(batch):
    question=[]
    pos_relations=[]
    neg_relations=[]
    triple_feather=[]



    for triple in batch:
        question.append(triple[0])
        pos_relations.append(triple[1])
        neg_relations.append(triple[2])


    encoded_input = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
    pos_input = tokenizer(pos_relations, padding=True, truncation=True, return_tensors='pt')
    neg_input = tokenizer(neg_relations, padding=True, truncation=True, return_tensors='pt')

    triple_feather.append(encoded_input)
    triple_feather.append(pos_input)
    triple_feather.append(neg_input)

    return  triple_feather


class DataLoaderChen(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderChen, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

if __name__=="__main__":
    train_data_path = 'train_demo.txt'
    test_data_path = "train_demo.txt"

    # train(train_data_path,test_data_path)

    question, poss, negs = load_data(train_data_path)