#!/bin/bash
exp='/username_data/dgx_expr/light_copy/experiments/bart-2stage2-bs6-ep6-3e-5/*'

for file in $exp
do
if [ -d "$file" ] 
then 
#   echo "$file is directory"
  if [ -f "$file/pytorch_model.bin" ]
  then
    echo "procssing $file/pytorch_model.bin"
    CUDA_VISIBLE_DEVICES=2,3 python finetune_trainer.py \
    --model_name_or_path ${file} \
    --tokenizer_name experiments/bart-2stage2-bs6-ep6-3e-5/ \
    --output_dir ${file}_val \
    --cache_dir /tmp/bart_cache \
    --task summarization \
    --num_train_epochs 8 \
    --data_dir /username_data/data_sum/cnn_dm \
    --do_eval \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 3e-5 \
    --warmup_step 500 \
    --save_steps 5000 \
    --predict_with_generate
  fi
fi
done
