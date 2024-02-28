import os, argparse, json

DATA = {
    'cnn': '/username_data/data_sum/cnn_dm',
    'xsum': '/username_data/data_sum/xsum',
    'nyt': '/username_data/data_sum/other_datasets/nyt',
    'bill': '/username_data/data_sum/other_datasets/billsum_v4_1/clean_final',
    'arkiv': '/username_data/data_sum/other_datasets/arxiv/arxiv-dataset',
    'tifu': '/username_data/data_sum/other_datasets/reddit_tifu',
    'wikihow': '/username_data/data_sum/other_datasets/wikihow'
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='/username_data/dgx_expr/light_copy/experiments/pretrain/mini_nat3_multi_bs80_ep4_1e-5/')
parser.add_argument('--run_path', type=str, default='/username_data/dgx_expr/light_copy/')
parser.add_argument('--cuda', type=str, default='-1')
parser.add_argument('--data', type=str, default='cnn')
parser.add_argument('--bs', type=str, default='2')
parser.add_argument('--save', type=str, default='res_all.json')
parser.add_argument('--all', action='store_true')
parser.add_argument('--last', action='store_true')
args = parser.parse_args()
# model_dir = '/username_data/dgx_expr/light_copy/experiments/pretrain/mini_nat3_multi_bs80_ep4_1e-5/'
# run_path = '/username_data/dgx_expr/light_copy'
device = ''
if args.cuda != "-1":
    device = "CUDA_VISIBLE_DEVICES=" + args.cuda
data = DATA[args.data]
ops = []
# os.system('cd ' + args.run_path)
cd = 'cd ' + args.run_path
if args.last:
    os.system('pwd')
    print("processing the last checkpoint...")
    last = device + ' python finetune_trainer.py --model_name_or_path ' \
        + args.model_dir \
        + ' --tokenizer_name ' \
        + args.model_dir \
        + ' --output_dir ' \
        + os.path.join(args.model_dir, 'transfer_'+args.data, 'last') \
        + ' --cache_dir /tmp/bart_cache --task summarization --data_dir ' \
        + data \
        + ' --do_predict --per_device_eval_batch_size ' \
        + args.bs \
        + ' --predict_with_generate'
    os.system(cd + '&& pwd &&' + last)
    ops.append(os.path.join(args.model_dir, 'transfer_'+args.data, 'last'))

if args.all:
    checkpoints_ = os.listdir(args.model_dir)
    checkpoints =[]
    for fn in checkpoints_:
        if not os.path.isfile(os.path.join(args.model_dir, fn)) and os.path.isfile(os.path.join(args.model_dir, fn, "pytorch_model.bin")):
            checkpoints.append(fn)
    print(checkpoints)

    for cp in checkpoints:
        print("processing checkpoints ", cp)
        cp_dir = os.path.join(args.model_dir, cp)
        cmd = device + ' python finetune_trainer.py --model_name_or_path ' \
            + cp_dir \
            + ' --tokenizer_name ' \
            + args.model_dir \
            + ' --output_dir ' \
            + os.path.join(args.model_dir, 'transfer_'+args.data, cp) \
            + ' --cache_dir /tmp/bart_cache --task summarization --data_dir ' \
            + data \
            + ' --do_predict --per_device_eval_batch_size ' \
            + args.bs \
            + ' --predict_with_generate'

        os.system(cd + '&& pwd &&' + cmd)
        ops.append(os.path.join(args.model_dir, 'transfer_'+args.data, cp))

    # all dicts
    res = {
        'name': [],
        'loss': [],
        'r1': [],
        'r2': [],
        'rl': [],
        'len': []
    }

    for op in ops:
        with open(op+'/test_results.json', 'r') as opf:
            r = json.load(opf)
        res['name'].append(os.path.split(op)[-1])
        res['loss'].append(r['test_loss'])
        res['r1'].append(r['test_rouge1'])
        res['r2'].append(r['test_rouge2'])
        res['rl'].append(r['test_rougeLsum'])
        res['len'].append(r['test_gen_len'])

    save_dir = os.path.dirname(args.save)
    if not os.path.exists(save_dir):
        os.system('mkdir ' + save_dir)
    with open(args.save, 'w') as f:
        json.dump(res, f, indent=4)

'''
#!/bin/bash
exp='/username_data/dgx_expr/light_copy/experiments/pretrain/mini_nat3_multi_bs80_ep4_1e-5/'

cd /username_data/dgx_expr/light_copy

echo "processing the last checkpoint"
CUDA_VISIBLE_DEVICES=8,9 python finetune_trainer.py \
    --model_name_or_path ${exp} \
    --tokenizer_name ${exp} \
    --output_dir ${exp}_last_cnn_test \
    --cache_dir /tmp/bart_cache \
    --task summarization \
    --num_train_epochs 8 \
    --data_dir /username_data/data_sum/cnn_dm \
    --do_predict \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
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
    CUDA_VISIBLE_DEVICES=8,9 python finetune_trainer.py \
    --model_name_or_path ${file} \
    --tokenizer_name ${exp} \
    --output_dir ${file}_cnn_test \
    --cache_dir /tmp/bart_cache \
    --task summarization \
    --num_train_epochs 8 \
    --data_dir /username_data/data_sum/cnn_dm \
    --do_predict \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 3e-5 \
    --warmup_step 500 \
    --save_steps 5000 \
    --predict_with_generate
  fi
fi
done
'''