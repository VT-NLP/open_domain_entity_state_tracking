#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os

import torch
import sys
import numpy as np
from tqdm import tqdm
import random
# from transformers import (
#     GPT2Config,
#     GPT2LMHeadModel,
#     GPT2Tokenizer,
# )

from gen_ans_to_list import aggregate_predictions
from pathlib import Path

# to avoid "src.xxx" not found error.
sys.path.insert(0, '..')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from extract_constraint import get_constraint_decoder
from gen_ans_to_list import aggregate_predictions


# MODEL_CLASSES = {
#     "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
# }

def get_ids(ground_entity):
    groud_id = ground_entity["input_ids"]
    G = []
    for id in groud_id:

        for word in id:
            s = [1, 3]
            if word not in s:
                G.append(word)
    return list(set(G))


class OpenPIGPT2Predictor:

    def __init__(self, model_path: str, stop_token: str = '<|endoftext|>'):
        set_seed(1)
        self.stop_token = stop_token
        self.tokenizer = AutoTokenizer.from_pretrained('t5-large')  # Fixed GPT2 tokenizer.
        # self.tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        self.tokenizer.add_tokens('[SN]')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.entities_path = os.getcwd() + "/node_relation_0_5.txt"
        self.entities = []
        with open(self.entities_path, "r") as fr:
            for line in fr.readlines():
                self.entities.append(line.strip())

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}

        self.train_attr_set = []
        attr_path = os.getcwd() + "/training_attr_set.txt"
        with open(attr_path, "r", encoding="utf-8") as fr_a:
            for line in fr_a.readlines():
                line = line.strip()
                self.train_attr_set.append(line)

        self.train_entity_set = []
        entity_path = os.getcwd() + "/training_entity_set.txt"
        with open(entity_path, "r", encoding="utf-8") as fr_e:
            for line2 in fr_e.readlines():
                line2 = line2.strip()
                self.train_entity_set.append(line2)


        def t5_dict_set():
            dic_ = {}
            with open(os.getcwd() +"/t5_dict_filter_right6.txt", "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    line = line.strip().split("\t")
                    word = line[0]
                    id = line[1]
                    if word not in dic_:
                        dic_[word] = id
            return dic_

        self.t5_dic = t5_dict_set().values()
        self.t5_dic = [int(x) for x in self.t5_dic]

        self.train_attr_set = get_ids(self.tokenizer(list(self.train_attr_set)))
        self.train_entity_set=get_ids(self.tokenizer(list(self.train_entity_set)))

        self.T5_dic = []
        for key, value in self.tokenizer.get_vocab().items():
            self.T5_dic.append(key)

        self.model.eval()
        logger.info(f"Loaded model for generation.")

    def get_points(self, knowledge, GE, GA, Max_length_entity, Max_length_attr):

        ht_r = {}
        for triples in knowledge:
            # eraser|RelatedTo-RelatedTo|hand|0.3662654161453247
            triples = triples.split("|")
            if len(triples) == 3:
                if triples[0] in self.entity_idxs:
                    head = self.entity_idxs[triples[0]]
                else:
                    head = 0

                if triples[1] in self.entity_idxs:
                    relation1 = self.entity_idxs[triples[1]]
                else:
                    relation1 = 0

                if triples[2] in self.entity_idxs:
                    tail1 = self.entity_idxs[triples[2]]
                else:
                    tail1 = 0

                ht_r[str(head) + "|" + str(tail1)] = str(relation1)

        entity_nodes_id = []
        attr_nodes_id = []
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

        # if len(entity_nodes_id) < Max_length_entity:
        #     entity_nodes_id = entity_nodes_id + [0] * (Max_length_entity - len(entity_nodes_id))
        # else:
        #     flags = random.sample(range(0, len(entity_nodes_id)), Max_length_entity)
        #     entity_nodes_id = np.array(entity_nodes_id)[flags]
        #
        # if len(attr_nodes_id) < Max_length_attr:
        #     attr_nodes_id = attr_nodes_id + [0] * (Max_length_attr - len(attr_nodes_id))
        # else:
        #     flags = random.sample(range(0, len(attr_nodes_id)), Max_length_attr)
        #     attr_nodes_id = np.array(attr_nodes_id)[flags]

        # print("entity_nodes_id",entity_nodes_id)
        # print("attr_nodes_id",attr_nodes_id)
        nodes = [x for x in entity_nodes_id] + [x for x in attr_nodes_id]
        # print("nodes", len(nodes))
        entity_nodes_id_index = [nodes.index(x) for x in entity_nodes_id]
        attr_nodes_id_index = [nodes.index(x) for x in attr_nodes_id]
        adjacency_matrix = []
        # all_relations=[]
        for i in nodes:
            sub_adj = []
            for j in nodes:
                if str(i) + "|" + str(j) in ht_r:
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

        after_edge_indices = []

        flag = []
        edge_type = []
        start_sub_edge_indice = []
        end_sub_edge_indice = []
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

        return nodes, after_edge_indices, edge_type, entity_nodes_id_index, attr_nodes_id_index

    def get_predictions(self, max_len, input_ctxt_and_query,
                        graph,
                        graph_entity,
                        graph_attr,
                        score_entity,
                        digital_,
                        temperature: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 0.9,
                        do_sample: bool = True,
                        num_return_sequences: int = 1):
        '''
        :param max_len: max number of tokens to generate overall.
        :param  top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.
                Must be between 0 and 1. Default to 0.9
        :param  temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        :param  top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 0 and infinity. Defaults to 0
        :param  num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.
        :param do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False`.
        :return: generated next token.
        '''
        encoded_prompt = self.tokenizer.encode(input_ctxt_and_query, add_special_tokens=False, return_tensors='pt')
        encoded_prompt = encoded_prompt.to(self.device)

        e_nodes_batch = []
        e_edge_indices_batch = []
        e_edge_type_batch = []

        entites_nodes_ids = []
        attr_nodes_ids = []

        # max_edge_index_e = []
        # max_edge_index_a = []
        Kg_entity_length = 8000
        kg_attr_length = 450
        Graph = graph.split(",")
        GE = graph_entity.split(",")
        GA = graph_attr.split(",")

        g_entity_id = get_ids(self.tokenizer(GE))
        g_attr_id = get_ids(self.tokenizer(GA))
        e_nodes, e_edge_indices, e_edge_type, entity_nodes_id, attr_nodes_id = self.get_points(Graph, GE, GA,
                                                                                               Kg_entity_length,
                                                                                               kg_attr_length)

        # max_edge_index_e.append(len(e_edge_type))
        e_nodes_batch.append(e_nodes)
        e_edge_indices_batch.append(e_edge_indices)
        e_edge_type_batch.append(e_edge_type)
        entites_nodes_ids.append(entity_nodes_id)
        attr_nodes_ids.append(attr_nodes_id)

        e_nodes_batch = torch.tensor(e_nodes_batch).to(torch.long).to(self.device)
        e_edge_indices_batch = torch.tensor(e_edge_indices_batch).to(torch.long).to(self.device)
        e_edge_type_batch = torch.tensor(e_edge_type_batch).to(self.device)
        # print("e_nodes_batch",e_nodes_batch.size())
        # print("e_edge_indices_batch",e_edge_indices_batch.size())

        entites_nodes_ids = torch.tensor(entites_nodes_ids).to(torch.long).to(self.device)
        attr_nodes_ids = torch.tensor(attr_nodes_ids).to(torch.long).to(self.device)

        decoding_format = "tree"
        # train_and_test_q_kg_attr = list(set(self.train_attr_set + g_attr_id+self.t5_dic+score_entity))
        # train_and_test_q_kg_entity = list(set(self.train_entity_set + g_entity_id + self.t5_dic + score_entity))
        train_and_test_q_kg_attr =list( set(list(self.t5_dic+digital_)))
        train_and_test_q_kg_entity =list(set(list(self.t5_dic+digital_)))
        # print("m",len(train_and_test_q_kg_attr))
        constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                    filted_entity_KG_set=train_and_test_q_kg_entity,
                                                    filted_attr_KG_set=train_and_test_q_kg_attr,
                                                    decoding_schema=decoding_format,
                                                    source_prefix=None)

        answer = self.generate_nexttokens_for_sent(max_len=max_len,
                                                   text_so_far=input_ctxt_and_query,
                                                   encoded_prompt=encoded_prompt,
                                                   temperature=temperature,
                                                   top_k=top_k,
                                                   top_p=top_p,
                                                   do_sample=do_sample,
                                                   num_return_sequences=num_return_sequences,
                                                   e_nodes_batch=e_nodes_batch,
                                                   e_edge_indices_batch=e_edge_indices_batch,
                                                   e_edge_type_batch=e_edge_type_batch,
                                                   entity_nodes_id=entites_nodes_ids,
                                                   attr_nodes_id=attr_nodes_ids,
                                                   constraint_decoder=constraint_decoder
                                                   )
        return {"answer": answer}

    def generate_nexttokens_for_sent(self,
                                     max_len: int,
                                     text_so_far: str,
                                     encoded_prompt: torch.Tensor,
                                     temperature: float,
                                     top_k: int,
                                     top_p: float,
                                     do_sample: bool,

                                     e_nodes_batch,
                                     e_edge_indices_batch,
                                     e_edge_type_batch,
                                     entity_nodes_id,
                                     attr_nodes_id,
                                     constraint_decoder,
                                     num_return_sequences) -> str:
        '''
        :param text_so_far: text generated so far.
        :param encoded_prompt: `tf.Tensor` of `dtype=tf.int32` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `tf.Tensor` of shape `(1,)`.
        :return: generated next token.
        '''

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            # print("encoded_prompt",encoded_prompt)
            src_sentence = encoded_prompt[0]
            # print("src_sentence",src_sentence)
            # print("sent",sent)
            # print("--------inter-prefix_allowed_tokens_fn")
            return constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                          tgt_generated=sent)

        answer: str = ""
        with torch.no_grad():
            out = self.model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=encoded_prompt,
                max_length=min(1024, max_len + encoded_prompt.size(-1)),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                e_nodes_batch=e_nodes_batch,
                e_edge_indices_batch=e_edge_indices_batch,
                e_edge_type_batch=e_edge_type_batch,
                entity_nodes_id=entity_nodes_id,
                attr_nodes_id=attr_nodes_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

            for out_seq in out:
                text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True)
                # text = text[len(text_so_far):]
                # text = text[: text.find(self.stop_token) if self.stop_token else None]
                answer += text
                answer += '. '

        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
        required=True
    )
    parser.add_argument(
        "--max_len",
        default=20,
        type=int,
        help="model path",
        required=True
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        default='<|endoftext|>',
        help="model path",
    )
    parser.add_argument(
        "--test_input_file",
        default=None,
        type=str,
        help="jsonl file containing id (str) and question (str) keys",
        required=True
    )
    parser.add_argument(
        "--unformatted_outpath",
        default=None,
        type=str,
        help="path to store unformatted model predictions",
        required=True
    )
    parser.add_argument(
        "--formatted_outpath",
        default="",
        type=str,
        help="path to store formatted model predictions (i.e. turns a string answer to an array of state changes).",
    )

    args = parser.parse_args()

    if not args.model_path or not os.path.exists(args.model_path):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation model file/dir does not exist: {args.model_path}")
        return

    if not args.test_input_file or not os.path.exists(args.test_input_file):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation input file does not exist: {args.test_input_file}")
        return

    if not args.unformatted_outpath:
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation output file is empty: {args.unformatted_outpath}")
        return

    args.model_path = args.model_path.strip()
    args.unformatted_outpath = args.unformatted_outpath.strip()
    args.test_input_file = args.test_input_file.strip()

    print(f"Generation task, input = {args.test_input_file}, output = {args.unformatted_outpath} ...")

    predictor = OpenPIGPT2Predictor(model_path=args.model_path, stop_token=args.stop_token)

    test_input = []
    with open(args.test_input_file, 'r') as open_file:
        for line in open_file:
            test_input.append(json.loads(line))

    Score_ids={}
    with open(os.getcwd() +"/test_word_score.jsonl",encoding="utf-8") as f2:
        for line in f2:
            # al = al + 1
            # print("--------al", al)
            input_json = json.loads(line)
            id = input_json["id"]
            filtered_entities=input_json["filtered_entities"].split(",")

            candidate_ids=[]
            for score_id in filtered_entities:
                score_id=score_id.split("|")
                score=float(score_id[1])
                if score >0.2:
                    candidate_ids.append(int(score_id[0]))

            Score_ids[id]=candidate_ids

    Digital_DIC={}
    with open(os.getcwd() + "/test_sentence_digital_candidate.jsonl", encoding="utf-8") as fdigital:
        for line in fdigital:
            # al = al + 1
            # print("--------al", al)
            input_json = json.loads(line)
            id = input_json["id"]
            digital=input_json["dgital"]
            Digital_DIC[id]=digital



    with open(args.unformatted_outpath, 'w') as open_file:
        for item in tqdm(test_input):
            output = predictor.get_predictions(max_len=args.max_len,
                                               input_ctxt_and_query=item['question'],
                                               graph=item["graph"],
                                               graph_entity=item["graph_entity"],
                                               graph_attr=item["graph_attr"],
                                               score_entity=Score_ids[item["id"]],
                                               digital_=Digital_DIC[item["id"]]
                                               )

            output['id'] = item['id']
            json.dump(output, open_file)
            open_file.write('\n')

    formatted_fp = args.unformatted_outpath + ".formatted.jsonl" \
        if not args.formatted_outpath else args.formatted_outpath
    logger.info(f"Done generating. Aggregating and formatting to {formatted_fp}")
    aggregate_predictions(prediction_fp=args.unformatted_outpath,
                          out_fp=formatted_fp)


if __name__ == "__main__":
    main()