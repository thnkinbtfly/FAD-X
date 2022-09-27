import os

save_folder = "amazon_gen_15eps_fuse"
os.makedirs(save_folder, exist_ok=True)
all_langs = ['en', 'zh', 'ja', 'es', 'de']
major_lang = {'en': 'de', 'zh': 'en', 'ja': 'zh', 'es': 'en', 'de': 'en'}
for seed in [42]:
    for num_samples in [2000]:
        for fusion_of_stacks in [True]:
            for with_head in [False]:
                for tgt_lang in all_langs:
                    without_tgt_lang = [l for l in all_langs if l != tgt_lang]
                    ml = major_lang[tgt_lang]
                    without_ml_lang = [l for l in without_tgt_lang if l != ml]
                    lists = [
                        ([tgt_lang] + without_tgt_lang, [tgt_lang] + without_tgt_lang, tgt_lang),
                        ([ml] + without_ml_lang, [ml] + without_ml_lang, tgt_lang),
                    ]

                    for task_adapts, lang_adapts, train_datasets in lists:
                        if not fusion_of_stacks:
                            lang_adapts = lang_adapts[0:1]

                        test_name = f"amazon_fos{fusion_of_stacks}_seed{seed}_task{','.join(task_adapts)}lang{','.join(lang_adapts)}_head{with_head}_sample{num_samples}_eps15_{tgt_lang}"
                        task_adapters = ""
                        lang_adapters = ""
                        for ta in task_adapts:
                            if tgt_lang == ta:
                                task_adapters += f"amazon_{ta}_samples{num_samples}_eps15/{num_samples}/{ta}/amazon_reviews_multi_{ta} "
                            else:
                                task_adapters += f"amazon_{ta}_samples200000_eps15/{ta}/amazon_reviews_multi_{ta} "
                        for la in lang_adapts:
                            lang_adapters += f"{la}/wiki@ukp  "
                        lang_adapters += f"--fusion_of_stacks={fusion_of_stacks} "


                        script = f"""
            export PYTHONPATH=`pwd`
            export task_adapter="{task_adapters}"
            export lang_adapter="{lang_adapters}"
            export output_dir=amazon_fusion/{test_name}
            export task_setting="--lang {tgt_lang} --train_adapter=True --load_adapter ${{task_adapter}} --load_lang_adapter ${{lang_adapter}} --language fusion "
            python3 train_amazon_fusion.py --epochs 15 --num_samples {num_samples} --with_head={with_head} --output_dir=${{output_dir}} $task_setting
                        """
                        with open(f"{save_folder}/{test_name}.sh", 'w') as f:
                            f.write(script)