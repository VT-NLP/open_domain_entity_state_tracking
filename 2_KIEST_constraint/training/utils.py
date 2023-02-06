import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

# import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
# from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler
import random
from transformers import PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
"""
this file from https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/utils.py
"""
# try:
#     from fairseq.data.data_utils import batch_by_size
#
#     FAIRSEQ_AVAILABLE = True
# except (ImportError, ModuleNotFoundError):
#     FAIRSEQ_AVAILABLE = False


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".jsonl")
        # print("self.src_file",self.src_file)

        self.tgt_file = Path(data_dir).joinpath(type_path + ".jsonl")
        # self.len_file = Path(data_dir).joinpath(type_path + ".len")
        self.graph_file = Path(data_dir).joinpath(type_path + ".graph.tok")
        self.src_lens = self.get_char_lens(self.src_file)
        self.used_char_len = True

        self.entities_path = Path(data_dir).joinpath("node_relation_0_5" + ".txt")
        self.entities = []
        with open(self.entities_path, "r") as fr:
            for line in fr.readlines():
                self.entities.append(line.strip())

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}


        self.max_source_length = max_source_length
        # print("  self.max_source_length",  self.max_source_length)
        self.max_target_length = max_target_length
        # print("-- self.max_target_length ", self.max_target_length )

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        # dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        char_lens=[]
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                input_json = json.loads(line)
                char_lens.append(len(input_json["question"]))

        return char_lens# [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)


    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")



class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""


    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        # #id question   answer  ground_entity ground_attr graph graph_entity graph_attr


        source_line = self.prefix + json.loads(linecache.getline(str(self.src_file), index).rstrip("\n"))["question"]
        # print("source_line",source_line)
        tgt_line = json.loads(linecache.getline(str(self.src_file), index).rstrip("\n"))["answer"].replace(",","[SN]")+" [SN]"
        # print("--tgt_line-",tgt_line)
        graph = json.loads(linecache.getline(str(self.src_file), index).rstrip("\n"))["graph"]
        graph_entity = json.loads(linecache.getline(str(self.src_file), index).rstrip("\n"))["graph_entity"]
        graph_attr = json.loads(linecache.getline(str(self.src_file), index).rstrip("\n"))["graph_attr"]


        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line,
                "graph": graph,
                "graph_entity":graph_entity,
                "graph_attr": graph_attr,
                "id": index - 1}

    def get_points(self,knowledge,GE,GA,Max_length_entity,Max_length_attr):

        ht_r={}
        for triples in knowledge:
            # eraser|RelatedTo-RelatedTo|hand|0.3662654161453247
            triples = triples.split("|")
            if len(triples)==3:
                if triples[0] in self.entity_idxs:
                    head = self.entity_idxs[triples[0]]
                else:
                    head=0

                if triples[1] in self.entity_idxs:
                    relation1 = self.entity_idxs[triples[1]]
                else:
                    relation1 = 0

                if triples[2] in self.entity_idxs:
                    tail1 = self.entity_idxs[triples[2]]
                else:
                    tail1 = 0


                ht_r[str(head)+"|"+str(tail1)]=str(relation1)

        entity_nodes_id=[]
        attr_nodes_id=[]
        for ent in GE:
            if ent in self.entity_idxs:
                entity_nodes_id.append(self.entity_idxs[ent])
            else:
                entity_nodes_id.append(0)

        for enta in GA:
            if enta in self.entity_idxs:
                attr_nodes_id.append(self.entity_idxs[enta])
            else:
                attr_nodes_id.append(0)




        if len(entity_nodes_id) < Max_length_entity:
            entity_nodes_id = entity_nodes_id + [0] * (Max_length_entity - len(entity_nodes_id))
        else:
            flags = random.sample(range(0, len(entity_nodes_id)), Max_length_entity)
            entity_nodes_id = np.array(entity_nodes_id)[flags]


        if len(attr_nodes_id) < Max_length_attr:
            attr_nodes_id = attr_nodes_id + [0] * (Max_length_attr - len(attr_nodes_id))
        else:
            flags = random.sample(range(0, len(attr_nodes_id)), Max_length_attr)
            attr_nodes_id = np.array(attr_nodes_id)[flags]

        # print("entity_nodes_id",entity_nodes_id)
        # print("attr_nodes_id",attr_nodes_id)
        nodes = [x for x in entity_nodes_id] + [x for x in attr_nodes_id]
        # print("nodes", len(nodes))
        entity_nodes_id_index=[nodes.index(x) for x in entity_nodes_id]
        attr_nodes_id_index = [nodes.index(x) for x in attr_nodes_id]
        adjacency_matrix=[]
        # all_relations=[]
        for i in nodes:
            sub_adj=[]
            for j in nodes:
                if str(i)+"|"+str(j) in ht_r:
                    # all_relations.append(ht_r[str(i)+"|"+str(j)])
                    sub_adj.append(1)
                else:
                    sub_adj.append(0)

            adjacency_matrix.append(sub_adj)



        edge_indices = torch.tensor(adjacency_matrix).nonzero().t().contiguous()

        """
        tensor([[17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 26, 26, 26, 30, 35, 35, 35,
         35, 40, 40, 40, 40, 56, 56, 56, 56, 56, 60, 60, 60, 60, 60, 65, 65, 72,
         72, 72, 82, 82, 85, 85, 85, 92, 92, 92, 92, 92, 99, 99],
        [ 9, 14, 17, 36, 42, 48, 60, 35, 46, 60, 92, 56, 75, 84, 24, 18, 35, 48,
         75, 12, 24, 40, 98, 26, 56, 65, 72, 75, 12, 16, 17, 18, 62, 56, 65, 56,
         72, 85, 39, 82,  7, 72, 85, 12, 18, 49, 69, 92, 12, 63]])
        """

        after_edge_indices=[]

        flag = []
        edge_type = []
        start_sub_edge_indice=[]
        end_sub_edge_indice=[]
        for i in range(edge_indices.size(1)):
            start_points = edge_indices[0][i]
            end_points = edge_indices[1][i]
            start_sub_edge_indice.append(int(start_points))
            end_sub_edge_indice.append(int(end_points))

            two_ = str(start_points) + "|" + str(end_points)
            two_reverse = str(end_points) + "|" + str(start_points)
            flag.append(two_)
            if i == 0:
                edge_type.append(1)
            else:
                if two_ not in flag or two_reverse not in flag:
                    edge_type.append(1)
                if two_reverse in flag:
                    edge_type.append(0)

        # print("start_sub_edge_indice",len(start_sub_edge_indice))
        # print("end_sub_edge_indice",len(end_sub_edge_indice))
        after_edge_indices.append(start_sub_edge_indice)
        after_edge_indices.append(end_sub_edge_indice)

        return nodes,after_edge_indices,edge_type,entity_nodes_id_index,attr_nodes_id_index



    def generate_mask_matrix(self,points, triples):
        weights = []
        for i in range(len(points)):

            weight1 = []
            for j in range(len(points)):
                if points[i].split("|")[1] == points[j].split("|")[0]:
                    if points[i] + "|" + points[j].split("|")[1] in triples:
                        weight1.append(1)
                    else:
                        weight1.append(0)

                else:
                    weight1.append(0)
            weights.append(weight1)
        return weights

    def collate_fn(self, batch):
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data

        Kg_entity_length = 1000#batch_encoding["input_ids"].size(1)
        kg_attr_length=140
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        """
              for the first time,  do not use the graph knowledge, just use sequence information
              """
        e_nodes_batch=[]
        e_edge_indices_batch=[]
        e_edge_type_batch=[]


        max_edge_index_e=[]

        entites_nodes_ids=[]
        attr_nodes_ids = []
        for sub_batch in batch:
            """---for entity---------"""

            Graph = sub_batch["graph"].split(",")
            GE=sub_batch["graph_entity"].split(",")
            GA = sub_batch["graph_attr"].split(",")

            e_nodes,e_edge_indices,e_edge_type,entity_nodes_id,attr_nodes_id = self.get_points(Graph,GE,GA,Kg_entity_length,kg_attr_length)

            max_edge_index_e.append(len(e_edge_type))
            e_nodes_batch.append(e_nodes)
            e_edge_indices_batch.append(e_edge_indices)
            e_edge_type_batch.append(e_edge_type)
            entites_nodes_ids.append(entity_nodes_id)
            attr_nodes_ids.append(attr_nodes_id)
        #


        Max_e_edge = max(max_edge_index_e)
        #
        #
        end_e_edge_indices_batch=[]
        for tupple_e in e_edge_indices_batch:
            new_tuple=[]
            if len(tupple_e[0])<Max_e_edge:
                new_tuple.append(tupple_e[0]+[0]*(Max_e_edge-len(tupple_e[0])))
                new_tuple.append(tupple_e[1] + [0] * (Max_e_edge - len(tupple_e[1])))
            else:
                new_tuple.append(tupple_e[0])
                new_tuple.append(tupple_e[1])
        #     # print("mmmm",len(new_tuple))
            end_e_edge_indices_batch.append(new_tuple)

        end_e_edge_type_batch=[]
        for type_e in e_edge_type_batch:
            if len(type_e) < Max_e_edge:
                end_e_edge_type_batch.append(type_e+ [0] * (Max_e_edge - len(type_e)))
            else:
                end_e_edge_type_batch.append(type_e)


        batch_encoding["e_nodes_batch"] = e_nodes_batch
        batch_encoding["e_edge_indices_batch"] = end_e_edge_indices_batch
        batch_encoding["e_edge_type_batch"] = end_e_edge_type_batch

        batch_encoding["entity_nodes_id"] = entites_nodes_ids
        batch_encoding["attr_nodes_id"] = attr_nodes_ids

        return batch_encoding



    def generate_edge_tensors(self, batch):

        graphs_edges = []
        set_edges = {'d': 0, 'r': 1}
        for g in batch:

            g = g["src_graphs"]

            edge_index_1 = []
            edge_index_2 = []
            edges_types = []

            x = g.split()
            for e in x:
                e = e.replace('(', '').replace(')', '')
                e = e.split(',')
                e2, e1, l = e

                if int(e1) >= self.max_source_length:
                    continue
                if int(e2) >= self.max_source_length:
                    continue

                edge_index_1.append(int(e1))
                edge_index_2.append(int(e2))
                edges_types.append(set_edges[l])

            edges_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
            edges_types = torch.tensor(edges_types, dtype=torch.long)

            graphs_edges.append((edges_index, edges_types))

        return graphs_edges


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)

class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))

def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False



def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
