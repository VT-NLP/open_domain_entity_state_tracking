from typing import Dict

from transformers import AutoTokenizer


def list_dictionary(d, n_tab=-1):
    if isinstance(d, list):
        for i in d:
            list_dictionary(i, n_tab)
    elif isinstance(d, dict):
        n_tab += 1
        for key, value in d.items():
            if key == '<end>':
                print("{}{}".format(" " * n_tab, key))
            else:
                print("{}{}".format(" " * n_tab, key))
                list_dictionary(value, n_tab)
    else:
        print("{}{}".format("\t" * n_tab, d))


def print_tree(tree):
    list_dictionary(tree)


def get_label_name_tree(label_name_list, tokenizer, end_symbol='<end>'):
    sub_token_tree = dict()
    # print("label_name_list",label_name_list)
    label_tree = dict()
    for typename in label_name_list:
        after_tokenized = tokenizer.encode(typename, add_special_tokens=False)

        label_tree[typename] = after_tokenized
    # print("label_tree",label_tree.items())

    for _, sub_label_seq in label_tree.items():
        parent = sub_token_tree
        for value in sub_label_seq:
            if value not in parent:
                parent[value] = dict()
            parent = parent[value]

        parent[end_symbol] = None
    # print("sub_token_tree-----",sub_token_tree)
    return sub_token_tree


def get_entity_ids(filted_entity_KG_set, tokenizer):
    filted_KG_entity_ids=[]
    for word in filted_entity_KG_set:
        after_tokenized_id = tokenizer.encode(word, add_special_tokens=False)
        # print(after_tokenized_id[0])
        filted_KG_entity_ids.append(after_tokenized_id[0])
        # print(after_tokenized_id[0])
        # break
    # print(after_tokenized_id)
    # print(len(filted_entity_KG_set))
    # print(len(after_tokenized_id))


class PrefixTree:
    def __init__(self, label_name_list, tokenizer, end_symbol='<end>'):
        self.label_name_list = label_name_list
        self._tokenizer = tokenizer
        self.label_name_tree = get_label_name_tree(
            label_name_list, tokenizer, end_symbol)
        self._end_symbol = end_symbol

    def is_end_of_tree(self, tree: Dict):
        return len(tree) == 1 and self._end_symbol in tree


if __name__ == "__main__":
    event_subtype_name = [line.strip() for line in """Die
    Marry
    Divorce
    Injure
    Transfer-Ownership
    Transfer-Money
    Transport
    Start-Org
    Be-Born
    End-Org
    Declare-Bankruptcy
    Merge-Org
    Attack
    Demonstrate
    Meet
    Phone-Write
    Start-Position
    End-Position
    Nominate
    Elect
    Arrest-Jail
    Release-Parole
    Charge-Indict
    Trial-Hearing
    Sue
    Convict
    Sentence
    Fine
    Execute
    Extradite
    Acquit
    Pardon
    Appeal""".split('\n')]

    test_tokenizer = AutoTokenizer.from_pretrained('t5-small')

    # suptype_tree = get_label_name_tree(
    #     event_suptype_name, test_tokenizer)
    # # role_tree = get_label_name_tree(ACEEventMetaData.event_role_name)
    # print_tree(suptype_tree)
    # # print_tree(role_tree)
