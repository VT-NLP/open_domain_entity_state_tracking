from transformers import T5EncoderModel,AutoConfig
import  torch.nn as nn
import torch
class Bert_Model(nn.Module):
   def __init__(self, classes):
       super(Bert_Model, self).__init__()
       config = AutoConfig.from_pretrained("t5-base")
       setattr(config, 'adapter_dim', 256)
       self.bert = T5EncoderModel.from_pretrained('t5-base',config=config)
   
       # self.out = nn.Linear(self.bert.config.hidden_size, classes)
       self.loss = torch.nn.BCELoss(reduction='sum')
       self.dropout = nn.Dropout(0.1)
       self.label_smoothing=0
       # relu activation function
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


   def forward(self, input,attention_mask):
       fw=open("results.txt","w")

       question_embedding= self.bert(input)

       # fw.write(str(question_embedding))
       # print("question_embedding",question_embedding.size())
       # question_embedding= self.out(question_embedding)
       # topic_entity_embedding = self.topic_entity_embedding_matrix(topic_entity_id)
       # triple_representation = self.get_triple_representation(question_embedding, topic_entity_embedding)
       # x=self.applyNonLinear(triple_representation)
       # print("last_hidden_state",question_embedding["last_hidden_state"].size()) #torch.Size([32, 15, 768])
       # print("pooler_output", question_embedding["pooler_output"].size()) # torch.Size([32, 768])
       # apply softmax activation
       question_embedding=torch.mean(question_embedding["last_hidden_state"],dim=1)
       x = self.applyNonLinear(question_embedding)
       #question_embedding= self.softmax(x)
       question_embedding= torch.sigmoid(x)
       #print("-",question_embedding)
       
      

       return question_embedding

