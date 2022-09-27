import os

save_folder = "wikiann_higher"
os.makedirs(save_folder, exist_ok=True)
other_langs = """
en
fi
tr
vi
zh
ar
ka
id
""".split('\n')
other_langs = [l.strip() for l in other_langs if l]
low_langs = ['qu', 'ilo', 'xmf', 'tk', 'gn', 'mi', 'mhr', 'cdo']
major_lang = {'mi': 'id', 'ilo': 'id', 'xmf': 'ka', 'qu': 'en',
              'cdo': 'zh', 'gn': 'en', 'tk': 'tr', 'mhr': 'fi'}
all_langs = other_langs
model_name = 'bert-base-multilingual-cased'
for seed in [42, 43, 44, 45, 46]:
    for fusion_of_stacks in [True, False]:
        for tgt_lang in low_langs:
            without_tgt_langs = all_langs
            ml = major_lang[tgt_lang]
            without_ml_langs = [l for l in without_tgt_langs if l != ml]
            lists = [
                ([tgt_lang] + without_tgt_langs, [tgt_lang] + without_tgt_langs, tgt_lang), # LAll
                ([ml] + without_ml_langs, [ml] + without_ml_langs, tgt_lang), #LAll w/o tgt
            ]

            for task_adapts, lang_adapts, train_datasets in lists:
                if not fusion_of_stacks:
                    lang_adapts = lang_adapts[0:1]

                test_name = f"fos{fusion_of_stacks}_seed{seed}_task{','.join(task_adapts)}lang{','.join(lang_adapts)}_tgt{tgt_lang}"
                task_adapters = ""
                lang_adapters = ""
                dataset_name = 'wikiann'
                for ta in task_adapts:
                    task_adapters += f"ner_{ta}/{dataset_name} "
                for la in lang_adapts:
                    lang_adapters += f"{la}/wiki@ukp "
                lang_adapters += f"--fusion_of_stacks={fusion_of_stacks} "


                script = f"""
    export PYTHONPATH=`pwd`
    export HF_DATASETS_CACHE="/data2/hf_datasets"
    export TRANSFORMERS_CACHE="/data2/hf_transformers"
    export model_name={model_name}
    export dataset_name="wikiann"
    export train_bs=16
    export epochs=100
    export fp16=True
    export lr=1e-4

    export task_adapter="{task_adapters}"
    export lang_adapter="{lang_adapters}"
    export output_dir=ner_fusion/{test_name}
    export data_setting="--task_name ner --dataset_name $dataset_name --dataset_config_name {tgt_lang}"
    export task_setting="--seed {seed} --train_adapter=True --load_adapter ${{task_adapter}} --load_lang_adapter ${{lang_adapter}} --language fusion"
    export train_setting="--learning_rate $lr --fp16=$fp16 --evaluation_strategy=epoch --save_strategy=epoch --save_total_limit=3 --metric_for_best_model=f1 --greater_is_better=True --num_train_epochs=$epochs --per_device_train_batch_size $train_bs --model_name_or_path $model_name --output_dir $output_dir"
    python3 examples/pytorch/token-classification/run_ner_fusion.py --do_train=True --do_predict=True --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
    python3 examples/pytorch/token-classification/run_ner_fusion.py --do_train=False --do_predict=True --neptune_prefix best_ --load_best --per_device_eval_batch_size=64 $task_setting $train_setting $data_setting
                """
                with open(f"{save_folder}/{test_name}.sh", 'w') as f:
                    f.write(script)