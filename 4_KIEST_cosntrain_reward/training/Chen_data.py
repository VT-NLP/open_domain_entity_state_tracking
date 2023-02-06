#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
#from pytorch_lightning.utilities import rank_zero_info
from utils import Seq2SeqDataset, freeze_embeds,freeze_params,assert_all_frozen,label_smoothed_nll_loss
from parameters import  model_args
import random
import numpy as np
import sys
#from transformers.modeling_bart import shift_tokens_right
from transformers import  T5ForConditionalGeneration
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
# from evaluate_model import evaluate
import os
# from check_point_file import  _sorted_checkpoints,_rotate_checkpoints
# from answer_generation import  StateChange4Predictor

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    #MBartTokenizer,
logger
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

class Seq2SeqDataModule():
    def __init__(self): #__init__(self, hparams, **kwargs):
        # self.hparams=hparams
        self.dataset_class = Seq2SeqDataset
        self.sortish_sampler=False
        self.max_tokens_per_batch=False
        self.gpus=0
        self.train_batch_size=6
        self.eval_batch_size=6
        self.num_workers=4
        self.n_train=-1 # "# examples. -1 means use all."
        self.n_val=-1 # "# examples. -1 means use all."
        self.n_test=-1 #"# examples. -1 means use all."
        self.max_target_length=384
        self.val_max_target_length=384
        self.test_max_target_length=384
        self.max_source_length=384
        self.data_dir="data/formatted_for_gpt2"
        self.model_config_prefix=None  # what is it???
        # tokenizer=None
        # # self.model_name_or_path="t5-small"
        self.cache_dir=None
        # config=None
        # num_labels=None
        # model=None
        #
        # MODEL_MODES = {
        #     "base": AutoModel,
        #     "sequence-classification": AutoModelForSequenceClassification,
        #     "question-answering": AutoModelForQuestionAnswering,
        #     "pretraining": AutoModelForPreTraining,
        #     "token-classification": AutoModelForTokenClassification,
        #     "language-modeling": AutoModelWithLMHead,
        #     "summarization": AutoModelForSeq2SeqLM,
        #     "translation": AutoModelForSeq2SeqLM,
        #     "graph2text": AutoModelForSeq2SeqLM,
        # }
        # self.model_type = MODEL_MODES["graph2text"]
        # """------------token---seting------------"""
        n_observations_per_split = {
        "train": self.n_train,
        "val": self.n_val,
        "test": self.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        #
        self.target_lens = {
            "train": self.max_target_length,
            "val": self.val_max_target_length,
            "test": self.test_max_target_length,
        }
        # assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        # assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        #
        self.dataset_kwargs: dict = dict(
            data_dir=self.data_dir,
            max_source_length=self.max_source_length,
            prefix=self.model_config_prefix or "",
        )
        #
        # if tokenizer is None:
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base",
            cache_dir=self.cache_dir,
        )
        # self.tokenizer.pad_token = '[PAD]'
        # tokenizer.sep_token = '.'
        self.tokenizer.add_tokens('[SN]')




    def train_dataloader(self) -> DataLoader:
        # print("self.train_batch_size",self.train_batch_size)
        dataloader = self.get_dataloader("train", batch_size=6, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:

        return self.get_dataloader("val", batch_size=self.eval_batch_size)

    def test_dataloader(self) -> DataLoader:

        return self.get_dataloader("test", batch_size=self.eval_batch_size)



    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        ignore the function of  make_sortish_sampler or   make_dynamic_sampler
        """
        # print("-------data---")
        # print("type_path",type_path)
        dataset = self.get_dataset(type_path)
        # print("--len(da",len(dataset))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=None,
        )


    def get_dataset(self, type_path) -> Seq2SeqDataset:
        # self.hparams.data_dir---data
        n_obs = self.n_obs[type_path] # None
        # print("n_obs",n_obs)
        max_target_length = self.target_lens[type_path] # 384
        # print("max_target_length",max_target_length)

        dataset = Seq2SeqDataset(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )

        return dataset


if __name__=="__main__":
    # print(model_args())
    seq2seq_data=Seq2SeqDataModule()
    # seq2seq_data.train_model()
    train_data=seq2seq_data.train_dataloader()
    # print("--train--data",train_data)

    #
    for epoch in range(0, 1):
        for i, data in enumerate(train_data):

            print("------over")
            # print(data)
            # print("")


