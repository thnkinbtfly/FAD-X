import os

model = 'bert-base-multilingual-cased'

langs = """
cdo
mhr
mi
ilo
qu
xmf
tk
gn
"""
fp16 = True
for bs in [16]:
    for seed in range(42, 47):
        reduction_factor = '0.64' # 16/25
        for lang in langs.split('\n'):
            output_dir = f"ner_{lang}_reduc{reduction_factor}"
            if seed > 42:
                output_dir += f"_seed{seed}"

            lang = lang.strip()
            if not lang:
                continue
            script = f"""
        export PYTHONPATH=`pwd`
        export HF_DATASETS_CACHE="/data2/hf_datasets"
        export TRANSFORMERS_CACHE="/data2/hf_transformers"
        export model_name={model}
        export output_dir={output_dir}
        export src_lang={lang}
        export dataset_name="wikiann"
        export train_bs={bs}
        export epochs=100
        export fp16={fp16}
        export lr=1e-4

        export data_setting="--task_name ner --dataset_name $dataset_name --dataset_config_name ${{src_lang}}"
        export task_setting="--train_adapter=True --adapter_reduction_factor {reduction_factor} --load_lang_adapter ${{src_lang}}/wiki@ukp --language ${{src_lang}}"
        export train_setting="--seed {seed} --learning_rate $lr --fp16=$fp16 --evaluation_strategy=epoch --save_strategy=epoch --save_total_limit=3 --metric_for_best_model=f1 --greater_is_better=True --num_train_epochs=$epochs --per_device_train_batch_size $train_bs --model_name_or_path $model_name --output_dir $output_dir"

        python3 examples/pytorch/token-classification/run_ner.py --do_train=True --do_predict=True --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
        python3 examples/pytorch/token-classification/run_ner.py --do_train=False --do_predict=True --neptune_prefix best_ --load_best --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
        
            """
            save_folder = f"wikiann_larger"
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, f"{output_dir}.sh"), 'w') as f:
                f.write(script)