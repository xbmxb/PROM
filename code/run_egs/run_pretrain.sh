CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_trainer.py \
--model_name_or_path facebook/bart-large \
--tokenizer_name facebook/bart-large \
--output_dir experiments/pretrain/mini1025_multi_bs96_ep4_1e-6 \
--cache_dir /tmp/bart_cache \
--task summarization \
--num_train_epochs 4 \
--data_dir /username_data/160g_en/output/mini_1025 \
--do_eval --do_train \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--learning_rate 1e-6 \
--warmup_step 500 \
--save_steps 5000 \
--predict_with_generate \
--gradient_accumulation_steps 2

#full data
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13 python finetune_trainer.py \
--model_name_or_path facebook/bart-large \
--tokenizer_name facebook/bart-large \
--output_dir experiments/pretrain/shortall_bs84_ep2_1e-6 \
--cache_dir /tmp/bart_cache \
--task summarization \
--num_train_epochs 2 \
--data_dir /username_data/160g_en/output_short/allshort3 \
--do_predict --do_train \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 1e-6 \
--warmup_step 500 \
--save_steps 5000 \
--predict_with_generate \
--gradient_accumulation_steps 7

# lead
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python finetune_trainer.py \
--model_name_or_path /username_data/dgx_expr/light_copy/experiments/con_ft/mini_nat3_bs80_ep4_1e-6_con_mini_6k/ \
--tokenizer_name /username_data/dgx_expr/light_copy/experiments/con_pretrain/mini_nat3_bs80_ep4_1e-6_con_mini \
--output_dir experiments/pretrain/lead_bs84_ep2_1e-6 \
--cache_dir /tmp/bart_cache \
--task summarization \
--num_train_epochs 2 \
--data_dir /username_data/160g_en/output_lead/mini3/ \
--do_train \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 1e-6 \
--warmup_step 500 \
--save_steps 5000 \
--predict_with_generate \
--gradient_accumulation_steps 7
