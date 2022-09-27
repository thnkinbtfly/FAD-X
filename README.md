# FAD-X: Fusioning Pretrained Adapters for Fast Adaptation to Low-Resource Languages

```
pip install -e .
pip install -r examples/pytorch/token-classification/requirements.txt
```

The code is based on [adapter-transformers](https://github.com/adapter-hub/adapter-transformers).

## WikiAnn experiments
run following python files to generate sh files.
- S(t) : [submit_scripts/gen_wikiann.py](submit_scripts/gen_wikiann.py). Must be ran before the followings (to generate TAs)
- S(t) w/ param+ : [submit_scripts/gen_wikiann_more_larger.py](submit_scripts/gen_wikiann_more_larger.py)
- Fuse(L), Fuse(L-LAt) : [submit_scripts/gen_fused_wikiann.py](submit_scripts/gen_fused_wikiann.py)
- Fuse(L), Fuse(L-LAt) w/ ml (most resource-abundant) : [submit_scripts/gen_fused_wikiann_ml.py](submit_scripts/gen_fused_wikiann_ml.py)

## Amazon experiments
- S(t) : [submit_scripts/gen_amazon.py](submit_scripts/gen_amazon.py). Must be ran before the followings (to generate TAs)
- Fuse(L), Fuse(L-LAt) : [submit_scripts/gen_fused_amazon.py](submit_scripts/gen_fused_amazon.py)