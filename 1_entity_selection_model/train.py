from util_ import load_data,load_test_data1,DatasetChenQA,DataLoaderChen,TripletLossC
import  json
from model import TransBERT
from transformers import AdamW,get_linear_schedule_with_warmup
import  torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
# from sentence_transformers import util
import logging
import argparse
import  torch.nn as nn
import heapq
"""
https://pypi.org/project/sentence-transformers/
"""
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--model_path', type=str, default="bert-base-uncased", help='the model path from transformers')
parser.add_argument('--n_epochs', type=int, default=5, help='input batch size')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--num_warmup_steps', type=int, default=50, help='num_warmup_steps')
parser.add_argument('--patience', type=float, default=5, help='update times, up to it, down')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")


def sentence_id_mask(sentence):
  encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
  return encoded_input

def set_model_logger(file_name):
  '''
  Write logs to checkpoint and console
  '''

  logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=file_name,
    filemode='w'
  )
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


def TE_BERT(model, test_data):
  """
  this is the method to test our model
  """
  print("----------testing begin------------")
  right=0
  R = 0
  al = 0

  fw = open("formatted_for_gpt2_clean_filterd_entity_KG/train.jsonl", "w")
  for doub in tqdm(test_data):
    """
    q, pos, candidate
    """
    al=al+1
    question = doub[0]
    question_feather=sentence_id_mask(question)
    question_embedding=model(question_feather)
    pos = doub[1]
    candidate=doub[2]
    Dic_ = {}
    Dic_["id"] = doub[3]
    Dic_["question"] = question # +instruction+prompt
    Dic_["answer"] = doub[4]
    Dic_["ground_entity"]=",".join(pos)
    Dic_["entities_within_hop2"] = ",".join(candidate)


    right_labels=[]  # index
    for p in pos:
      if p in candidate:
        l=candidate.index(p)
        right_labels.append(l)

    with torch.no_grad():

      relation_feather = sentence_id_mask(candidate)
      relation_embedding = model(relation_feather)
      # print("relation_embedding",relation_embedding.size())
      question_embedding = question_embedding.repeat(relation_embedding.size(0), 1)
      # print(question_embedding)
      cos_sim = cos(question_embedding, relation_embedding)
      # end = [float(x) for x in d]
      # # print("--------",end)
      filter_entity_KG=[]
      predicts=[]
      f=-1
      for can_entity_score in cos_sim:
        f=f+1
        if can_entity_score>0.3:
          predicts.append(f)
          #print(can_entity_score.item())
          filter_entity_KG.append(str(can_entity_score.item())+"|"+str(candidate[f]))

      filter_entity_KG=",".join(filter_entity_KG)
      Dic_["filtered_entities"] =filter_entity_KG
      fw.writelines(json.dumps(Dic_) + "\n")
      fw.flush()


      hit=0
      for pre_score in predicts:
        if pre_score in right_labels:
          hit = hit + 1

      sub_all = len(pos)

      if sub_all != 0:
        recall = hit / sub_all
      else:
        recall = 0

      R = R + recall
    # R=0




  # print("the result is", right / len(test_data)) # becauce the all sample in the test is 1815, we can not /len(test_data)
  print("the final result is:",R/al)

  return R/al



def train(train_data_path,test_data_path):

  """----------load the train and test data----------"""
  question, poss, negs = load_data(train_data_path)

  T = load_test_data1(test_data_path)


  train_data_web = []
  for idx in range(len(question)):
    train_data_web.append([question[idx], poss[idx], negs[idx]])
  train_data_web = DatasetChenQA(train_data_web)

  train_dataloader = DataLoaderChen(train_data_web, shuffle=True, batch_size=args.batch_size)
  print(" the data load is over")

  """----------model initialization----------"""
  model=TransBERT()

  """----------optimizer and some schedulers----------"""
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)


  n_epochs = args.n_epochs
  total_steps = len(train_dataloader) * n_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,  # Default value in run_glue.py
                                              num_training_steps=total_steps)


  """----------define loss function----------"""
  LossF=TripletLossC()
  model.zero_grad() # this step is vert important,!!!! we must zero the weight of the model in the first
  if torch.cuda.is_available():
    model.cuda()

  patience = args.patience
  no_update = 0
  best_model = model.state_dict()
  best_score = -float("inf")


  """----------ok let's train it----------"""
  fw=open("results.txt","w",encoding="utf-8")
  for epoch in range(n_epochs):
    print("the current epoch is:",epoch)
    model.train()  # this step is vert important  !!!!
    train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
    for step, batch in enumerate(train_dataloader):
      question_feather = batch[0]
      pos_relation = batch[1]
      neg_relation = batch[2]

      question_embedding=model(question_feather) # if we want to calculate the question feather, input yes
      pos_embedding=model(pos_relation)
      neg_embedding=model(neg_relation)

      loss=LossF(question_embedding,pos_embedding,neg_embedding)
      # print(loss)


      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

    """----------ok let's test it----------"""
    model.eval()
    global_results = TE_BERT(model, T)
    fw.write("global_results----"+str(global_results)+"\n")

  #
    eps = 0.0001
    if global_results > best_score + eps:
      best_score = global_results
      no_update = 0
      best_model = model.state_dict()
      logging.info(" accuracy %s increased from previous epoch" % (str(global_results)))
      # global_results=  validate(model=model, data_path= valid_data_path, word2idx= word2ix,device=device)
      logging.info('Test global accuracy %s for best valid so far:' % (str(global_results)))
      # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
      suffix = ''
      checkpoint_path = 'checkpoints/'
      checkpoint_file_name = checkpoint_path + suffix + ".pt"
      logging.info('Saving checkpoint to %s' % checkpoint_file_name)
      torch.save(model.state_dict(), checkpoint_file_name)
    elif (global_results < best_score + eps) and (no_update < patience):
      no_update += 1
      logging.info("Validation accuracy decreases to %s from %s, %d more epoch to check" % (
        global_results, best_score, patience - no_update))

    elif no_update == patience:
      logging.info("Model has exceed patience. Saving best model and exiting")
      torch.save(best_model, checkpoint_path + "best_score_model.pt")
      exit()
    if epoch == n_epochs - 1:
      logging.info("Final Epoch has reached. Stopping and saving model.")
      torch.save(best_model, checkpoint_path + "best_score_model.pt")
      exit()



if __name__=="__main__":
  train_data_path='formatted_for_gpt2_clean_entity_knowledge/train.jsonl'
  test_data_path='formatted_for_gpt2_clean_entity_knowledge/train.jsonl'
  #
  # train_data_path='final_dataset/train_q_ground_cand.jsonl'
  # test_data_path="final_dataset/test_q_ground_cand.jsonl"

  train(train_data_path,test_data_path)

  # question, poss, negs = load_data(train_data_path)

