import os

model = 'bert-base-multilingual-cased'

hrls = """
en
fi
tr
vi
zh
ar
ka
id

de
fr
ru
es
ja
""".split('\n')
hrls = [l.strip() for l in hrls if l]

lrls = """
ilo
qu
xmf
tk
gn
cdo
mhr
mi
""".split('\n')
lrls = [l.strip() for l in lrls if l]

langs = hrls + lrls
fp16 = True
for bs in [16]:
    for seed in range(42, 47):
        for lang in langs:
            output_dir = f"ner_{lang}"
            if seed > 42:
                if lang in hrls:
                    continue
                output_dir += f"_seed{seed}"

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
        export task_setting="--train_adapter=True --load_lang_adapter ${{src_lang}}/wiki@ukp --language ${{src_lang}}"
        export train_setting="--seed {seed} --learning_rate $lr --fp16=$fp16 --evaluation_strategy=epoch --save_strategy=epoch --save_total_limit=3 --metric_for_best_model=f1 --greater_is_better=True --num_train_epochs=$epochs --per_device_train_batch_size $train_bs --model_name_or_path $model_name --output_dir $output_dir"

        python3 examples/pytorch/token-classification/run_ner.py --do_train=True --do_predict=True --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
        python3 examples/pytorch/token-classification/run_ner.py --do_train=False --do_predict=True --neptune_prefix best_ --load_best --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
            """
            save_folder = f"wikiann_madx"
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, f"{output_dir}.sh"), 'w') as f:
                f.write(script)