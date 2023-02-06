
from transformers import  AdamW
from transformers import T5Tokenizer,AutoConfig
import numpy as np
tokenizer = T5Tokenizer.from_pretrained('t5-base', do_lower_case=True)
import random

from transformers import T5EncoderModel
import  torch.nn as nn
import torch
class Bert_Model(nn.Module):
   def __init__(self, classes):
       super(Bert_Model, self).__init__()
       config = AutoConfig.from_pretrained("t5-base")
       setattr(config, 'adapter_dim', 256)
       self.bert = T5EncoderModel.from_pretrained('t5-base',config=config)
      

       self.loss = torch.nn.BCELoss(reduction='sum')
       self.dropout = nn.Dropout(0.1)
       self.label_smoothing=0
       self.relu = nn.ReLU()

       # dense layer 1
       # self.fc1 = nn.Linear(768, 512)

       W1 = torch.ones(400, 768)
       W1 = nn.init.uniform_(W1)
       self.W1 = nn.Parameter(W1)
       self.fc1 = nn.Linear(768, 512)
       # dense layer 2 (Output layer)
       self.fc2 = nn.Linear(512, 2)

       # softmax activation function
       self.softmax =torch.nn.Softmax(dim=1)

   def applyNonLinear(self, question_embedding):
       x = self.fc1(question_embedding)
       x = self.relu(x)
       x = self.dropout(x)
       # output layer
       x = self.fc2(x)
       return x


   def forward(self, input):
       fw=open("results.txt","w")

       question_embedding= self.bert(input)
       question_embedding=torch.mean(question_embedding["last_hidden_state"],dim=1)
       x = self.applyNonLinear(question_embedding)
       question_embedding= self.softmax(x)
       return question_embedding


def R_c_model():
    patience = 5
    classes = 2
    model = Bert_Model(classes)

    model.cuda()
    best_model = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # args.adam_epsilon  - default is 1e-8. )

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Create the learning rate scheduler.
    fname = '/home/mingchen/01_RewardModel3_seed10_test1/training/classify_checkpoints/best_score_model.pt'
    model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))

    return model

