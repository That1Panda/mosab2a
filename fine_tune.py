
import os

import subprocess

def finetune_model(base_model_name, tokenized_name, corpus_directory):

    # Install required packages
    packages = [
        'torch',
        'transformers',
        'sentencepiece==0.1.97',
        'pytorch-lightning==1.9.5',
        'deepspeed==0.9.2',
        'wandb',
        'ninja',
        'rwkv==0.7.4',
        'lm_dataformat',
        'tqdm',
        'ftfy',
        'jsonlines'
    ]
    pip_install_cmd = f'pip install {" ".join(packages)}'
    subprocess.run(pip_install_cmd.split())
    import jsonlines

    # Set model directory
    model_dir = '/RWKV_model'
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)

    # Download base model
    base_model_url = f"https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/{base_model_name}"
    base_model_filename = os.path.basename(base_model_url)
    base_model_download_cmd = f'wget {base_model_url}'
    subprocess.run(base_model_download_cmd.split())

    # Clone RWKV-LM repository
    force_clone = True
    if force_clone:
        subprocess.run(['rm', '-rf', 'RWKV-LM'])
        subprocess.run(['rm', '-rf', 'json2binidx_tool'])
    subprocess.run(['git', 'clone', 'https://github.com/BlinkDL/RWKV-LM'])
    subprocess.run(['git', 'clone', '-b', 'rwkv-tokenizer', 'https://github.com/cahya-wirawan/json2binidx_tool'])

    # # Preprocess data
    # def get_files(folder, suffix=".txt", subdirs=True):
    #     file_dirs = []
    #     if subdirs:
    #         for root, dirs, files in os.walk(folder):
    #             for file in files:
    #                 if file.endswith(suffix):
    #                     file_dir = os.path.join(root, file)
    #                     file_dirs.append(file_dir)
    #     else:
    #         for file in os.listdir(folder):
    #             if file.endswith(suffix):
    #                 file_dir = os.path.join(folder, file)
    #                 file_dirs.append(file_dir)
    #     return file_dirs

    input_files = ','.join(data)
    preprocess_cmd = f'python ./json2binidx_tool/tools/preprocess_data.py --input {input_files} --output-prefix {tokenized_name} --vocab ./json2binidx_tool/rwkv_vocab_v20230424.txt --dataset-impl mmap --tokenizer-type RWKVTokenizer --append-eod'
    subprocess.run(preprocess_cmd.split())

    # Fine-tune the model
    ctxlen = 1024
    n_layer = 24
    n_embd = 1024
    precision = "bf16"
    finetune_cmd = f'python "./RWKV-LM/RWKV-v4neo/train.py" --load_model "../../{base_model_filename}" --proj_dir "../../finetuned" --data_file "../../{tokenized_name}_text_document" --data_type "binidx" --vocab_size 65536 --ctx_len {ctxlen} --epoch_steps 16 --epoch_count 200 --epoch_begin 0 --epoch_save 20 --micro_bsz 2 --n_layer {n_layer} --n_embd {n_embd} --pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 5e-6 --warmup_steps 50 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 --precision {precision} --strategy deepspeed_stage_2 --grad_cp 0'
    subprocess.run(finetune_cmd.split())

    # Unmount Google Drive
    drive_unmount_cmd = 'umount /content/drive'
    subprocess.run(drive_unmount_cmd.split())

# Usage
base_model_name = "RWKV-4-World-7B-v1-20230626-ctx4096.pth"
tokenized_name = "tokenized"
data = 'output.txt'

finetune_model(base_model_name, tokenized_name, data)
