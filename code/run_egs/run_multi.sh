CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_trainer.py \
--model_name_or_path facebook/bart-large \
--tokenizer_name facebook/bart-large \
--output_dir experiments/bart-multi_a100_bs6_3e-5 \
--cache_dir /tmp/bart_cache \
--task summarization \
--num_train_epochs 8 \
--data_dir /username_data/data_sum/cnn_dm \
--do_predict --do_train \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--learning_rate 3e-5 \
--warmup_step 500 \
--save_steps 5000 \
--predict_with_generate

