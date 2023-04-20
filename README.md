##  This is the source code for open_domain_entity_state_tracking

###  package version

NVIdIa Quadro rTX 8000 48G

python 3.6.5

torch version: 1.4.0+cu100

transformers :4.1.1

### 1-Data Progress
1) to correct the data in OpenPI (Data_progress/0_Data_Correction)
2) how to retrival the knowledge in the ConcpetNet in hop-N (Data_progress/1_get_kowledge_from_KG)
3) to create two selections (entity selection, attribute selection) (Data_progress/2_selection_model)
4) to build the  entity-attribute knowledge graph by selected entities/attribute knowledge (Data_progress/3_combine_entity_attr_to_graph)
5) others (how to get the entity/attribute embedding) in step 4).  (Data_progress4_get_E_embedding)
Note: above file is in the Google Driver [Data progress](https://drive.google.com/file/d/1aNgYVn039msTHOdjKAL0NI8nwY__z82x/view?usp=share_link)
###  2-Entity and Attr Selection Building
In  this section, a pre-trained model  with the triple loss function is used to create the entity and attr selection. the source code is in [selection model](https://drive.google.com/file/d/1OItH-PH0SMG-RCiX4mWKJBh4ZlMawRUr/view?usp=share_link).
please uses the default parameters in the scripts. How to run? In entity or attr selection model, please run:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 3-KIEST w/o ESC
This is the variants of KIEST without the reward.
Full model, please check [KIEST w/o ESC](https://drive.google.com/file/d/1KRPXbiUXOoDWCOvgsHZ9pLPp3Cd8vGLo/view?usp=share_link)

**3.1-Train**: please enter the 2_KIEST_constraint and run:
```
CUDA_VISIBLE_DEVICES=2 python training/run_trainer.py --output_dir=training_output --model_type=t5-large --continue_from_dir=/continue_from_dir --model_name_or_path=t5-large --do_train --train_data_file=data/formatted_for_gpt2/train.jsonl --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --overwrite_output_dir --length 100 --block_size 300 --save_total_limit 3 --save_steps 5000 --learning_rate 0.00005 --overridden_model_configs '{"resid_pdrop": 0.1, "attn_dropout": 0.1}' --weight_decay 0.0 --num_train_epochs 50 --seed 42
```
Note: the trained checkpoinks file is in training_output.

**3.2-Generation**:
Please get into the 2_KIEST_constraint file  and run,
```
bash scripts/predictions_bash.sh
```
you can change your own paramters in the predictions_bash.sh. The generated file 2_KIEST_constraint\data\prediction_format.jsonl is same as the  file test_our_model\prediction_format_wo_ESC.jsonl

**3.3-Evaluation**： please get into 6_test_our_model, and run:

```
python 1_change_file.py
python 2_simple_eval.py
```

**3.4-Results**：

<img src="https://github.com/VT-NLP/open_domain_entity_state_tracking/blob/main/with_reward.png" width="300"/>
Note: the BLEU is the BLEU-2, ROUGE is the ROUGE-L.

### 4-Classification Reward:
get into the 3_reward_clasification and run:

```
CUDA_VISIBLE_DEVICES=0 python train.py
```
### 5-KIEST:
Full files please check [4_KIEST_cosntrain_reward](https://drive.google.com/file/d/1CtWmN0q5gSLGiN2Q3WRW8JxNBggOf8j0/view?usp=share_link)

**5.1-Train**: please enter the 4_KIEST_cosntrain_reward and run:
```
CUDA_VISIBLE_DEVICES=1 python training/run_trainer.py --output_dir=training_output --model_type=t5-large --continue_from_dir=/continue_from_dir --model_name_or_path=t5-large --do_train --train_data_file=data/formatted_for_gpt2/train.jsonl --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --overwrite_output_dir --length 100 --block_size 300 --save_total_limit 3 --save_steps 5000 --learning_rate 0.00005 --overridden_model_configs '{"resid_pdrop": 0.1, "attn_dropout": 0.1}' --weight_decay 0.0 --num_train_epochs 10 --seed 10
```
Note: the trained checkpoinks file is in training_output.

**5.2-Generation**:
Please get into the 4_KIEST_cosntrain_reward file  and run,
```
bash scripts/predictions_bash.sh
```

**5.3-Evaluation**： please get into 6_test_our_model, and run:

```
python 1_change_file.py
python 2_simple_eval.py (the input file is prediction_format_constraint_reward_define_0.1.jsonl)
```

**5.4-Results**：

<img src="https://github.com/VT-NLP/open_domain_entity_state_tracking/blob/main/final.png" width="300"/>
Note: the BLEU is the BLEU-2, ROUGE is the ROUGE-L.
