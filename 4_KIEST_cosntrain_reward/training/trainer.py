import glob
import os
import random
import re
import shutil
import sys
from typing import List, Tuple
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from parameters import  model_args
from transformers import PreTrainedModel, PreTrainedTokenizer, AdamW, get_linear_schedule_with_warmup, logger,T5ForConditionalGeneration
from random import  sample
from pick_model import evaluate
from Chen_data import  Seq2SeqDataModule
# import sys
# sys.path.append("..")
from reward_classification_model import  R_c_model
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# import R_c_model
# Seq2SeqDataModule
# R_c_model.cuda().eval()
seq2seq_data=Seq2SeqDataModule()

def get_SN_index(list):
    SN_index=[0]
    SN_index_tuple=[]
    for i in range(len(list)):
        if list[i]==32100:
            SN_index.append(i)
    # print(SN_index)
    for j in range(len(SN_index)):
        if len(SN_index[j:j+2])==2:
            IND = SN_index[j:j + 2]
            SN_index_tuple.append(IND)
    return SN_index_tuple
def sample_3d(probs, temperature=1):
    '''probs.shape = (batch, seq_len, dim)
    ([6, 12, 32101]
    '''
    sample_idx = torch.zeros(probs.size(0), probs.size(1)).cuda()#.to(device)
    sample_probs = torch.zeros(probs.size(0), probs.size(1)).cuda()#.to(device)
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    # print("-------temp",temp.size())
    for i, s in enumerate(temp):
        # print("s",s.size()) #torch.Size([12, 50265]) #
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1) 作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_probs[i] = temp_probs.squeeze(1)

    return sample_probs, sample_idx.long()

def classify_reward(tgt_cls,sample_probs):
    tgt_reward = tgt_cls[:, 0] - tgt_cls[:, 1]
    sample_probs = sample_probs.contiguous()
    sample_logprobs = torch.log(sample_probs)
    reward = tgt_reward.unsqueeze(1).contiguous()
    output = -sample_logprobs * reward
    output_loss = output.mean()

    return output_loss


def cal_bl_reward(inp, tgt):
    '''Caculate BLEU-based reward'''
    smooth = SmoothingFunction()
    bleus = []
    for hyp, ref in zip(inp, tgt):
        bleus.append(sentence_bleu([ref], hyp,
                                   smoothing_function=smooth.method1))
    bleus = torch.FloatTensor(bleus).cuda()

    return bleus

def train(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
        # tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log", "train"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = seq2seq_data.train_dataloader()
    test_dataloader = seq2seq_data.test_dataloader()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  # Added here for reproducibility


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], file=sys.stdout, mininterval=10)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            pad_token_id = tokenizer.pad_token_id
            # inputs, labels, _ = batch
            src_ids=batch["input_ids"].to(args.device)
            src_mask =  batch["attention_mask"].to(args.device)
            tgt_ids = batch["labels"].to(args.device)

            e_nodes_batch=torch.tensor(batch["e_nodes_batch"]).to(torch.long).to(args.device)
            e_edge_indices_batch=torch.tensor(batch["e_edge_indices_batch"]).to(args.device)
            e_edge_type_batch=torch.tensor(batch["e_edge_type_batch"] ).to(args.device)
            entity_nodes_id=torch.tensor(batch["entity_nodes_id"]).to(torch.long).to(args.device)
            attr_nodes_id=torch.tensor(batch["attr_nodes_id"]).to(args.device)

            if isinstance(model, T5ForConditionalGeneration):
                # print("-----fuck this world----")
                decoder_input_ids = model._shift_right(tgt_ids)
            else:
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

            model.train()
            # outputs = model(inputs, labels=labels)
            outputs=model(src_ids,
                          attention_mask=src_mask,
                          decoder_input_ids=decoder_input_ids,

                          e_nodes_batch=e_nodes_batch,
                          e_edge_indices_batch=e_edge_indices_batch,
                          e_edge_type_batch=e_edge_type_batch,

                          entity_nodes_id=entity_nodes_id,
                          attr_nodes_id=attr_nodes_id,
                          use_cache=False
                       )


            lm_logits = outputs[0]
            loss_rewards=0
            blue_rewards = 0
            if len(decoder_input_ids)>1:
                results_sample = sample(list(range(len(decoder_input_ids))), 1)
            else:
                results_sample = sample(list(range(len(decoder_input_ids))), 1)


            for _id in results_sample: # choose some question (many tuples) in a batch
                # print("decoder_input_ids",decoder_input_ids.size()) #torch.Size([6, 222])
                # print("lm_logits", lm_logits.size()) # ([6, 222, 32101])
                # print("decoder_input_ids[_id]",decoder_input_ids[_id].size()) #([222])
                results = get_SN_index(decoder_input_ids[_id])  # many tuples

                if len(results)>1:
                    results=sample(results,1)
                else:
                    results = sample(results, 1)

                f=-1
                for re_index in results:
                    f=f+1
                    start=re_index[0]
                    end=re_index[1]
                    tgt_ref =decoder_input_ids[_id][start+1:end].unsqueeze(0)

                    out = lm_logits[_id, start+1:end, :].unsqueeze(0)
                    # print("-------out",out.size())
                    out = F.softmax(out, dim=-1)
                    sample_probs, sample_idx = sample_3d(out)
                    # print("sample_idx",sample_idx.size())
                    C_model=R_c_model()
                    tgt_cls = C_model(sample_idx).detach()
                    output_loss=classify_reward(tgt_cls, sample_probs)
                    loss_rewards = loss_rewards + output_loss

                    greedy_probs, greedy_idx = torch.max(out, dim=-1)
                    tgt_ref=[[int(x) for x in list(tgt_ref[0])]]
                    greedy_idx = [[int(x) for x in list(greedy_idx[0])]]
                    sample_idx = [[int(x) for x in list(sample_idx[0])]]

                    tgt_sam = cal_bl_reward(sample_idx, tgt_ref)
                    tgt_gre = cal_bl_reward(greedy_idx, tgt_ref)
                    # print("tgt_sam",tgt_sam)
                    # print("tgt_gre",tgt_gre)
                    sample_probs = sample_probs.contiguous()
                    sample_logprobs = torch.log(sample_probs)
                    output = -sample_logprobs * (tgt_gre - tgt_sam) * 0.2
                    output = output.mean()
                    blue_rewards=blue_rewards+output


            if args.label_smoothing == 0:
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
                loss = 0.9*ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.contiguous().view(-1))+0.1*loss_rewards#+blue_rewards

            else:
                lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
                )


            # print("loss",loss)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            # print("tr_loss",tr_loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.continue_from_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
