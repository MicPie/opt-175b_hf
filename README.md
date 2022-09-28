# OPT-175B for Huggingface

The OPT-175B for Huggingface setup is based on [Minimal OPT](https://github.com/zphang/minimal-opt) (especially https://github.com/zphang/minimal-opt/blob/main/minimal_opt/hf_conversion.py).

***This repo is still in development and I'll run soon some more tests for verification.***

However, if you are interested and you want to play around feel free and give feedback! :-)

# Preprocess weights:

1. Adapt `base_path` in `merged_shards_to_mmap.py` and run the file to create mmap files for each weight.
1. Adapt `base_path` in `mmap_to_pytorch.py` and run the file to create bin files for each weight and `pytorch_model.bin.index.json`.
1. Copy the `config.json` to your `layer_bins` directory.
1. Load model with (the slightly different max memory was needed to get it loaded, but this needs further investigation):
```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
            base_path_bins,
            device_map="auto",
            max_length=1024,
            load_in_8bit=True,
            max_memory = {
            0: "48GB",
            1: "48GB",
            2: "48GB",
            3: "45GB",
            }
        )
```
