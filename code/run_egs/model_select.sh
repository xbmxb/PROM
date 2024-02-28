#!/bin/bash
exp='/username_data/dgx_expr/light_copy/experiments/bart-multi_a100_bs6a2_3e-5/'

echo "procssing last "
CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune_trainer.py \
--model_name_or_path ${exp} \
--tokenizer_name ${exp} \
--output_dir ${exp}last_res \
--cache_dir /tmp/bart_cache \
--task summarization \
--num_train_epochs 8 \
--data_dir /username_data/data_sum/cnn_dm \
--do_predict \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--learning_rate 3e-5 \
--warmup_step 500 \
--save_steps 5000 \
--predict_with_generate

for file in $exp*
do
if [ -d "$file" ] 
then 
#   echo "$file is directory"
  if [ -f "$file/pytorch_model.bin" ]
  then
    echo "procssing $file/pytorch_model.bin"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune_trainer.py \
    --model_name_or_path ${file} \
    --tokenizer_name ${exp} \
    --output_dir ${file}_res \
    --cache_dir /tmp/bart_cache \
    --task summarization \
    --num_train_epochs 8 \
    --data_dir /username_data/data_sum/cnn_dm \
    --do_predict \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --warmup_step 500 \
    --save_steps 5000 \
    --predict_with_generate
  fi
fi
done
