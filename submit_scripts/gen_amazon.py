import os

langs = """
en
zh
de
ja
es
"""
fp16 = True
for num_samples in [2000, 200000]:
    for lang in langs.split('\n'):
        output_dir = f"amazon_{lang}_samples{num_samples}_eps15"
        lang = lang.strip()
        if not lang:
            continue
        script = f"""
    export PYTHONPATH=`pwd`
    export HF_DATASETS_CACHE="/data2/hf_datasets"
    export TRANSFORMERS_CACHE="/data2/hf_transformers"
    python train_amazon.py --epochs 15 --num_samples {num_samples} --lang {lang} --output_dir {output_dir}
        """
        save_folder = f"amazon_base_eps15"
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, f"{output_dir}.sh"), 'w') as f:
            f.write(script)