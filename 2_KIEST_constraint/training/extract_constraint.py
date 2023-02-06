from typing import List, Dict
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


def get_constraint_decoder(tokenizer, filted_entity_KG_set,filted_attr_KG_set,other_position_set, decoding_schema, source_prefix=None):
    if decoding_schema == 'tree':
        return TreeConstraintDecoder(tokenizer=tokenizer, filted_entity_KG_set=filted_entity_KG_set,
                                     filted_attr_KG_set=filted_attr_KG_set,
                                     other_position_set=other_position_set,
                                     source_prefix=source_prefix)


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
    def __init__(self, tokenizer, filted_entity_KG_set,filted_attr_KG_set,other_position_set, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.filted_entity_KG_set=filted_entity_KG_set
        self.filted_attr_KG_set=filted_attr_KG_set
        self.other_position_set=other_position_set
        # print(  self.filted_entity_KG_set)
        """
        "self.type_tree",self.type_tree {28596: {60: {18: {21347: {9433: {75: {63: {'<tree-end>': None}}}}}}}, 1193: {7287: {17: {'<tree-end>': None}}}}
        """
    def check_state(self, tgt_generated):
        # print("2----enter-------------check_state")
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1
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
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))
        if len(special_index_token) != 0:
            last_special_index, last_special_token = special_index_token[-1]

            if len(special_index_token) == 1:
                if last_special_token != 13:
                    return 'error', 0

            of_was_position = find_bracket_position(
                tgt_generated, _type_start=13, _type_end=47)

            begin_of_position = find_bracket_position(
                tgt_generated, _type_start=32100, _type_end=13)
            # {13: [1, 13], 47: [5]}
            # print("of_was_position", of_was_position)
            start_number, end_number = len(of_was_position[13]), len(
                of_was_position[47])

            a_start_number, a_end_number = len(begin_of_position[32100]), len(
                begin_of_position[13])


            if start_number == end_number and start_number != 0 and end_number != 0:
                state = 'generate_after_was'
            elif start_number == end_number + 1:
                state = 'generate_entity'
            elif start_number == 0:
                state = 'no_constraint_continue_generate_before_of'
            elif a_start_number==a_end_number:
                state='generate_attr'

            else:
                state = 'error'

            return state, last_special_index
        else:
            last_special_index, last_special_token = None, None
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
        state, index = self.check_state(tgt_generated)
        if state == 'start':
            valid_tokens = self.other_position_set#list(self.tokenizer.get_vocab().values())  # None#[self.type_start]
            return valid_tokens
        elif state == 'generate_entity':
            return  self.filted_entity_KG_set
        elif state=="generate_attr":
            return self.filted_attr_KG_set
        else:
            valid_token = self.other_position_set#list(self.tokenizer.get_vocab().values())  # [self.tokenizer.eos_token_id]
            return valid_token
