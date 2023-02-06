import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  AdamW
from transformers import T5Tokenizer
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,recall_score,f1_score
tokenizer = T5Tokenizer.from_pretrained('t5-base', do_lower_case=True)
import torch.nn as nn
import time
import datetime
import random
from model import Bert_Model
import json
from tqdm import  tqdm
MAX_LEN = 140


def labeltoOneHot(label):
    if label == "0":
        return [0, 1]
    else:  # indices == "y"
        return [1, 0]


def get_data(file):
    sentences = []
    labels = []
    with open(file, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.strip().split("\t")

            sentences.append(line[0])
            labels.append(labeltoOneHot(line[1]))

    return sentences, labels


def get_input_and_mask(sentences):
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    # Print sentence 0, now as a list of IDs.

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return input_ids, attention_masks


def get_input_id():
    train_sentences, train_labels = get_data("data/reward_classification_train.txt")

    train_input_ids, train_attention_masks = get_input_and_mask(train_sentences)

    test_sentences, test_labels = get_data("data/reward_classification_test.txt")
    test_input_ids, test_attention_masks = get_input_and_mask(test_sentences)

    train_inputs = torch.tensor(train_input_ids)
    test_inputs = torch.tensor(test_input_ids)

    train_labels = torch.FloatTensor(train_labels)
    test_labels = torch.FloatTensor(test_labels)
    #
    train_masks = torch.tensor(train_attention_masks)
    test_masks = torch.tensor(test_attention_masks)

    batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader


def flat_PRF1(preds, labels):
    pred_flat = preds
    labels_flat = labels
    P=accuracy_score(pred_flat, labels_flat)
    R=recall_score(pred_flat, labels_flat)
    F1=f1_score(pred_flat, labels_flat)
    return P,R,F1



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def TE(model, validation_dataloader):
    model.eval()
    # Tracking variables
    test_loss, test_accuracy ,test_f1,test_recall= 0, 0,0,0
    nb_test_steps, nb_test_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # speeding up validation
        with torch.no_grad():
            outputs, _ = model(input=b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs
        logits = torch.argmax(logits, dim=1)
        b_labels = torch.argmax(b_labels, dim=1)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        P, R, F1=flat_PRF1(logits, label_ids)

        # Accumulate the total accuracy.
        test_accuracy += P
        test_f1+=F1
        test_recall+=R
        # Track the number of batches
        nb_test_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(test_accuracy / nb_test_steps))
    print("  Recall: {0:.2f}".format(test_recall / nb_test_steps))
    print("  F1: {0:.2f}".format(test_f1 / nb_test_steps))
    print("  test took: {:}".format(format_time(time.time() - t0)))

    global_P = test_accuracy / nb_test_steps
    global_R = test_recall / nb_test_steps
    global_F1 = test_f1 / nb_test_steps

    return global_P,global_R,global_F1


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def preprocess_entities_relations(entity_dict, entities):
    e = {}

    f = open(entity_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        ent_id = int(line[0])
        ent_name = line[1].lower()
        e[ent_name] = entities[ent_id]
    f.close()

    return e


if __name__ == "__main__":
    patience = 7
    classes = 2


    time_start = time.time()

    train_dataloader, test_dataloader = get_input_id()

    model = Bert_Model(classes)
    # Tell pytorch to run this model on the GPU.
    model.cuda()
    best_model = model.state_dict()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  # args.adam_epsilon  - default is 1e-8. )

    # Create the learning rate scheduler.
    epochs = 5
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    # For each epoch...
    best_score = -float("inf")

    model.zero_grad()
    fw_results = open("global_results.txt", "w")
    for epoch_i in tqdm(range(0, epochs)):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].cuda()  # to(device)
            b_input_mask = batch[1].cuda()  # to(device)
            b_labels = batch[2].cuda()  # to(device)

            score, loss = model(input=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels
                                )

            #

            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        #
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        eps = 0.0001
        print("testing---")
        global_P,global_R,global_F1 = TE(model, test_dataloader)
        test_per="P:"+str(global_P) +"--R:"+str(global_R)+"-F1:"+str(global_F1)
        print("----training-----")
        train_global_P, train_global_R, train_global_F1 = TE(model, train_dataloader)
        train_per = "P:" + str(train_global_P) + "--R:" + str(train_global_R) + "-F1:" + str(train_global_F1)

        fw_results.write(train_per+"\t"+test_per+ "\n")
        fw_results.flush()

        if global_F1 > best_score:
            best_score = global_F1
            no_update = 0
        #     best_model = model.state_dict()
        #     print(" F1 %s increased from previous epoch" % (str(global_F1)))
            checkpoint_path = 'checkpoints/'
            torch.save(best_model, checkpoint_path + "best_score_model.pt")
        #     checkpoint_file_name = checkpoint_path + ".pt"
        #     torch.save(model.state_dict(), checkpoint_file_name)
        # #
        elif (global_F1 < best_score + eps) and (no_update < patience):
            no_update += 1
            print("Validation F1 decreases to %s from %s, %d more epoch to check" % (
                global_F1, best_score, patience - no_update))
        elif no_update == patience:
            print("Model has exceed patience. Saving best model and exiting")
            # torch.save(best_model, checkpoint_path + "best_score_model.pt")
            time_end = time.time()
            fw_results.write("time-cost:" + str((time_end - time_start) / 60) + "\n")
            exit()

        if epoch_i == epochs - 1:
            print("Final Epoch has reached. Stopping and saving model.")
            # torch.save(best_model, checkpoint_path + "best_score_model.pt")
            time_end = time.time()
            fw_results.write("time-cost:" + str((time_end - time_start) / 60) + "\n")
            exit()


