import os
import sys
from collections import OrderedDict
from tqdm.auto import tqdm
import json
import numpy as np
import glob
import torch


base_path = ""
base_path_mmap = base_path+"layer_mmaps/"
base_path_bins = base_path+"layer_bins/"


# create weight shapes dictionary
sys.path.append(base_path)  # the base_path is also where the py files should be located

from merged_shards_to_mmap import (
        vocab_size,
        num_attention_heads,
        hidden_size,
        ffn_dim,
        max_position_embeddings,
        num_hidden_layers,
        dims_per_head,
        )

weight_shapes = {
        "lm_head.weight": (vocab_size, hidden_size),
        "model.decoder.embed_tokens.weight": (vocab_size, hidden_size),
        "model.decoder.embed_positions.weight": (max_position_embeddings+2, hidden_size),
        "model.decoder.final_layer_norm.weight": (hidden_size,),
        "model.decoder.final_layer_norm.bias": (hidden_size,),
        "model.decoder.final_layernorm.weight": (hidden_size,),
        "model.decoder.final_layernorm.bias": (hidden_size,),
        }

for layer_i in range(num_hidden_layers):
    layer = {
            f"model.decoder.layers.{layer_i}.self_attn.k_proj.weight": (num_attention_heads, dims_per_head, hidden_size),
            f"model.decoder.layers.{layer_i}.self_attn.v_proj.weight": (num_attention_heads, dims_per_head, hidden_size),
            f"model.decoder.layers.{layer_i}.self_attn.q_proj.weight": (num_attention_heads, dims_per_head, hidden_size),
            f"model.decoder.layers.{layer_i}.self_attn.k_proj.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.self_attn.v_proj.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.self_attn.q_proj.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.self_attn.out_proj.weight": (hidden_size, num_attention_heads, dims_per_head),
            f"model.decoder.layers.{layer_i}.self_attn.out_proj.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.self_attn_layer_norm.weight": (hidden_size,),
            f"model.decoder.layers.{layer_i}.self_attn_layer_norm.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.fc1.weight": (ffn_dim, hidden_size),
            f"model.decoder.layers.{layer_i}.fc1.bias": (ffn_dim,),
            f"model.decoder.layers.{layer_i}.fc2.weight": (hidden_size, ffn_dim),
            f"model.decoder.layers.{layer_i}.fc2.bias": (hidden_size,),
            f"model.decoder.layers.{layer_i}.final_layer_norm.weight": (hidden_size,),
            f"model.decoder.layers.{layer_i}.final_layer_norm.bias": (hidden_size,),
            }
    weight_shapes = {**weight_shapes, **layer}


# export mmaps to separate pytorch weights and HF json description
weight_map = {}
fns = sorted(glob.glob(base_path_mmap+"/*.mmap"))
for fn in (pbar := tqdm(fns)):
    weight_name = fn.split("/")[-1].split(".mmap")[0]
    pbar.set_description(weight_name)

    weight_data = np.memmap(fn, shape=weight_shapes[weight_name], dtype=np.float16, mode="r")

    torch_dict = OrderedDict()
    torch_dict[weight_name] = torch.tensor(weight_data).half()

    fn_out = fn.replace("layer_mmaps", "layer_bins").replace(".mmap", ".bin")

    torch.save(torch_dict, fn_out)
    weight_map[weight_name] = fn_out


fn_json_out = base_path_bins+"pytorch_model.bin.index.json"

json_out = {
        "metadata": {
            "total_size": sum(os.path.getsize(base_path_bins+f) for f in os.listdir(base_path_bins))
                    },
        "weight_map": weight_map
        }

with open(fn_json_out, "w") as f:
        json.dump(json_out, f, indent=4, sort_keys=True)
