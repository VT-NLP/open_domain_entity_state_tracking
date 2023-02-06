

#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict

# from data_convert.format.text2tree import type_start, type_end
from label_tree import get_label_name_tree, get_entity_ids

import os

debug = True if 'DEBUG' in os.environ else False
debug_step = True if 'DEBUG_STEP' in os.environ else False


def match_sublist(the_list, to_match):
    """

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position


def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):
    print(generated, src_sequence) if debug else None

    if len(generated) == 0:
        # It has not been generated yet. All SRC are valid.
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


def get_constraint_decoder(tokenizer, filted_entity_KG_set,train_and_test_q_kg_attr, decoding_schema, source_prefix=None):
    if decoding_schema == 'tree':
        return TreeConstraintDecoder(tokenizer=tokenizer, filted_entity_KG_set=filted_entity_KG_set,
                                     train_and_test_q_kg_attr=train_and_test_q_kg_attr,
                                     source_prefix=source_prefix)
    # elif decoding_schema == 'treespan':
    #     return SpanConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    # else:
    #     raise NotImplementedError(
    #         'Type Schema %s, Decoding Schema %s do not map to constraint decoder.' % (
    #             decoding_schema, decoding_schema)
    #     )


class ConstraintDecoder:
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        if debug:
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))

        valid_token_ids = self.get_state_valid_tokens(
            src_sentence.tolist(),
            tgt_generated.tolist()
        )

        if debug:
            print('========================================')
            print('valid tokens:', self.tokenizer.convert_ids_to_tokens(
                valid_token_ids), valid_token_ids)
            if debug_step:
                input()

        # return self.tokenizer.convert_tokens_to_ids(valid_tokens)
        # print("========",valid_token_ids)
        return valid_token_ids


class TreeConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, filted_entity_KG_set,train_and_test_q_kg_attr, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        # print("type_schema",type_schema.type_list)
        self.entity_tree = get_label_name_tree(
            filted_entity_KG_set, self.tokenizer, end_symbol=self.tree_end)

        """
        "self.type_tree",self.type_tree {28596: {60: {18: {21347: {9433: {75: {63: {'<tree-end>': None}}}}}}}, 1193: {7287: {17: {'<tree-end>': None}}}}
        """

        self.attr_tree = get_label_name_tree(
            train_and_test_q_kg_attr, self.tokenizer, end_symbol=self.tree_end)
        print(self.attr_tree)

    def check_state(self, tgt_generated):
        # print("2----enter-------------check_state")

        """
        13--of
        47--was
        274--before
        15627--afterwards
         6--,
         1--</s>
         [32100]   [SN]

        """
        special_token_set = {13, 47, 32100}
        # print("--special_token_set", special_token_set)
        # tgt_generated = [32098, 32099, 4, 5, 6, 32098, 32099]
        # print("tgt_generated",tgt_generated)
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        # print("------special_index_token", special_index_token)
        # [(1, 13), (5, 47), (11, 32100), (13, 13)]
        # print("-----------",special_index_token)
        if len(special_index_token) != 0:
            last_special_index, last_special_token = special_index_token[-1]

            if len(special_index_token) == 1:
                if last_special_token != 13:
                    return 'error', 0

            of_was_position = find_bracket_position(
                tgt_generated, _type_start=13, _type_end=47)
            # {13: [1, 13], 47: [5]}
            # print("of_was_position", of_was_position)
            start_number, end_number = len(of_was_position[13]), len(
                of_was_position[47])

            if start_number == end_number and start_number != 0 and end_number != 0:
                state = 'generate_after_was'
            elif start_number == end_number + 1:
                state = 'generate_entity'
            # elif start_number == 0 and tgt_generated[-1] != self.tokenizer.pad_token_id:
            #     state = 'no_constraint_continue_generate_before_of'
            # elif start_number == 0 and end_number == 0:
            #     state = 'no_constraint_continue_generate'

            else:
                state = 'error'

            return state, last_special_index
        else:
            # return "no_constraint_continue_generate_before_of", None
            if tgt_generated[-1] == self.tokenizer.pad_token_id:
                return 'start', -1
            else:
                return "no_constraint_continue_generate_before_of", None



    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        # print("enter-------------search_prefix_tree_and_sequence")
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())

        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """
        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(
                self.tokenizer.eos_token_id)]
            # print("tokenizer---src_sentence",src_sentence) # [3845, 92, 293, 7, 3, 9, 4390, 11, 3, 9, 2252, 2478, 11, 3, 9, 8468, 3, 5, 1]

        state, index = self.check_state(tgt_generated)

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]
            return valid_tokens

        if state == 'start':
            # valid_tokens = list(self.tokenizer.get_vocab().values())#None#[self.type_start]
            attr_Tokens = list(self.attr_tree.keys())
            # print("Tokens",Tokens)
            if "<tree-end>" in attr_Tokens:
                attr_Tokens.pop(attr_Tokens.index("<tree-end>"))
                if len(attr_Tokens) == 0:
                    attr_Tokens = [self.tokenizer.eos_token_id]
                return attr_Tokens  # list(self.tokenizer.get_vocab().values())#Tokens
            else:
                return attr_Tokens
            # return valid_tokens

        elif state == 'generate_entity':
            # valid_tokens = list(self.tokenizer.get_vocab().values())  # None#[self.type_start]
            # return valid_tokens
            if tgt_generated[-1] == 13:
                # Start Event Label
                Tokens=list(self.entity_tree.keys())
                #print("Tokens",Tokens)
                if "<tree-end>" in Tokens:
                    Tokens.pop(Tokens.index("<tree-end>"))
                    if len(Tokens)==0:
                        Tokens=[self.tokenizer.eos_token_id]
                    return  Tokens#list(self.tokenizer.get_vocab().values())#Tokens
                else:
                    return Tokens#list(self.tokenizer.get_vocab().values())#Tokens
                    #else:
                        #return list(self.tokenizer.get_vocab().values())


                # list(self.entity_tree.keys())
            else:
            # print("index",index)
                if tgt_generated[-1] not in self.entity_tree:
                    return list(self.tokenizer.get_vocab().values())  # None
                else:
                    sub_tree = self.entity_tree[tgt_generated[-1]]
                    # print("-sub_tree",sub_tree)
                    is_tree_end = len(sub_tree) == 1 and self.tree_end in sub_tree
                    if is_tree_end:
                        valid_token = [47]  # was---47
                        return valid_token
                    #
                    if self.tree_end in sub_tree and len(sub_tree) > 1:
                        keys_ = list(sub_tree.keys())
                        if "<tree-end>" in keys_:
                            keys_.pop(keys_.index("<tree-end>"))
                            valid_token = keys_ + [47]
                            return valid_token
                        else:
                            valid_token = keys_
                            return valid_token

                # valid_token = list(self.tokenizer.get_vocab().values())  # [self.tokenizer.eos_token_id]
                # return valid_token
        # elif state == 'no_constraint_continue_generate_before_of':
        #     if tgt_generated[-1] not in self.attr_tree:
        #         return list(self.tokenizer.get_vocab().values())  # None
        #
        #     else:
        #         sub_attr_tree = self.attr_tree[tgt_generated[-1]]
        #         # print("-sub_tree",sub_tree)
        #         is_tree_end = len(sub_attr_tree) == 1 and self.tree_end in sub_attr_tree
        #         if is_tree_end:
        #             valid_token = [13]  # of---13
        #             return valid_token
        #         #
        #         if self.tree_end in sub_attr_tree and len(sub_attr_tree) > 1:
        #             keys_ = list(sub_attr_tree.keys())
        #             if "<tree-end>" in keys_:
        #                 keys_.pop(keys_.index("<tree-end>"))
        #                 valid_token = keys_ + [13]
        #                 return valid_token
        #             else:
        #                 valid_token = keys_
        #                 return valid_token

        elif state == 'generate_after_was':
            if tgt_generated[-1] !=32100:
                return list(self.tokenizer.get_vocab().values())  # None
            else:
                if tgt_generated[-1] not in self.attr_tree:
                    return list(self.tokenizer.get_vocab().values())  # None

                else:
                    sub_attr_tree = self.attr_tree[tgt_generated[-1]]
                    # print("-sub_tree",sub_tree)
                    is_tree_end = len(sub_attr_tree) == 1 and self.tree_end in sub_attr_tree
                    if is_tree_end:
                        valid_token = [13]  # was---13
                        return valid_token
                    #
                    if self.tree_end in sub_attr_tree and len(sub_attr_tree) > 1:
                        keys_ = list(sub_attr_tree.keys())
                        if "<tree-end>" in keys_:
                            keys_.pop(keys_.index("<tree-end>"))
                            valid_token = keys_ + [13]
                            return valid_token
                        else:
                            valid_token = keys_
                            return valid_token

        else:
            valid_token = list(self.tokenizer.get_vocab().values())  # [self.tokenizer.eos_token_id]
            return valid_token
