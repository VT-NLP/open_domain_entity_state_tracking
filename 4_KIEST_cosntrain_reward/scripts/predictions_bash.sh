TRAINED_MODEL=${1-"training_output"} 
INPUT_TO_PRED_CSV=${2-"data/formatted_for_gpt2/test.jsonl"}
OUTPUT_FILEPATH_CSV=${3-"data/prediction.jsonl"}
AGG_OUTPUT_FILEPATH_CSV=${4-"data/prediction_format.jsonl"}
MAX_LEN=${5-400}

set -x  # print the command being executed.

IFS=',' read -ra prediction_input_files <<< "$INPUT_TO_PRED_CSV"
IFS=',' read -ra prediction_output_files <<< "$OUTPUT_FILEPATH_CSV"
IFS=',' read -ra agg_pred_output_files <<< "$AGG_OUTPUT_FILEPATH_CSV"

for i in ${!prediction_input_files[*]}; do
  CUDA_VISIBLE_DEVICES=0 python training/generation.py \
      --model_path "$TRAINED_MODEL" \
      --test_input_file "${prediction_input_files[i]}" \
      --unformatted_outpath "${prediction_output_files[i]}" \
      --formatted_outpath "${agg_pred_output_files[i]}" \
      --max_len $MAX_LEN
done