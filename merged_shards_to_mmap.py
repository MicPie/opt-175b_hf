# This script is heavily based on: https://github.com/zphang/minimal-opt/blob/main/minimal_opt/hf_conversion.py

from tqdm.auto import tqdm
import numpy as np
import torch


base_path = ""
base_path_mmap = base_path+"layer_mmaps/"


# sharded checkpoint files should be located in the base_path directory
sharded_checkpoint_list = [
        "reshard-model_part-0-shard0.pt",
        "reshard-model_part-1-shard0.pt",
        "reshard-model_part-2-shard0.pt",
        "reshard-model_part-3-shard0.pt",
        "reshard-model_part-4-shard0.pt",
        "reshard-model_part-5-shard0.pt",
        "reshard-model_part-6-shard0.pt",
        "reshard-model_part-7-shard0.pt"
        ]


# OPT-175B configuration
vocab_size = 50272
num_attention_heads = 96
hidden_size = 12288
ffn_dim = 49152
max_position_embeddings = 2048
num_hidden_layers = 96
dims_per_head = hidden_size // num_attention_heads


def take_out(flat_params, shape):
    return flat_params[:np.prod(shape)].view(*shape), flat_params[np.prod(shape):]


def get_slice(shard_size, shard_i):
    return slice(shard_size * shard_i, shard_size * (shard_i + 1))


def create_mmap_file(name, base_path, shape, dtype='float16', mode='w+'):
    return np.memmap(base_path+"layer_mmaps/"+name+".mmap",  dtype=dtype, mode=mode, shape=shape)


def load_sharded_weights(base_path, sharded_checkpoint_list):
    """Load sharded weights and save to mmap file."""

    num_shards = len(sharded_checkpoint_list)

    # noinspection PyUnresolvedReferences
    vocab_size_per_shard = vocab_size // num_shards
    heads_per_shard = num_attention_heads // num_shards
    hidden_size_per_shard = hidden_size // num_shards
    ffn_dim_per_shard = ffn_dim // num_shards
    dims_per_head = hidden_size // num_attention_heads

    # mmap dicts for each layer
    model_decoder_layers_self_attn_k_proj_weight = {}
    model_decoder_layers_self_attn_v_proj_weight = {}
    model_decoder_layers_self_attn_q_proj_weight = {}

    model_decoder_layers_self_attn_k_proj_bias = {}
    model_decoder_layers_self_attn_v_proj_bias = {}
    model_decoder_layers_self_attn_q_proj_bias = {}

    model_decoder_layers_self_attn_out_proj_weight = {}
    model_decoder_layers_self_attn_out_proj_bias = {}

    model_decoder_layers_self_attn_layer_norm_weight = {}
    model_decoder_layers_self_attn_layer_norm_bias = {}

    model_decoder_layers_fc1_weight = {}
    model_decoder_layers_fc1_bias = {}

    model_decoder_layers_fc2_weight = {}
    model_decoder_layers_fc2_bias = {}

    model_decoder_layers_final_layer_norm_weight = {}
    model_decoder_layers_final_layer_norm_bias = {}


    for shard_i in (pbar := tqdm(range(num_shards))):
        loaded = torch.load(base_path+sharded_checkpoint_list[shard_i], map_location="cpu")
        if len(loaded["model"]) == 2:
            # small_model
            flat_params = loaded["model"]["flat_param_0"]
            load_final_layer_norm_first = False
        else:
            # big model
            load_final_layer_norm_first = True
            flat_params = torch.cat([
                v.flatten()
                for k, v in loaded["model"].items()
                if k != "decoder.version"
            ])

        # Vocab
        print(shard_i, "lm_head.weight")
        if shard_i == 0:
            lm_head_weight = create_mmap_file(
                    "lm_head.weight",
                    base_path,
                    shape=(vocab_size, hidden_size),
                    )
            model_decoder_embed_tokens_weight = create_mmap_file(
                    "model.decoder.embed_tokens.weight",
                    base_path,
                    shape=(vocab_size, hidden_size),
                    )
        out, flat_params = take_out(flat_params, (vocab_size_per_shard, hidden_size))
        model_decoder_embed_tokens_weight[get_slice(vocab_size_per_shard, shard_i)] = out.numpy()
        model_decoder_embed_tokens_weight.flush()
        lm_head_weight = model_decoder_embed_tokens_weight
        lm_head_weight.flush()


        # Pos encoding (fixed offset=2)
        print(shard_i, "model.decoder.embed_positions")
        if shard_i == 0:
            model_decoder_embed_positions_weight = create_mmap_file(
                "model.decoder.embed_positions.weight",
                base_path,
                shape=(max_position_embeddings+2, hidden_size),
                )
        out, flat_params = take_out(flat_params, (max_position_embeddings + 2, hidden_size))
        model_decoder_embed_positions_weight[:] = out.numpy()
        model_decoder_embed_positions_weight.flush()


        if load_final_layer_norm_first:
            # Post-attention LayerNorm
            print(shard_i, "model.decoder.final_layer_norm")
            if shard_i == 0:
                model_decoder_final_layer_norm_weight = create_mmap_file(
                        "model.decoder.final_layer_norm.weight",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_final_layer_norm_bias = create_mmap_file(
                        "model.decoder.final_layer_norm.bias",
                        base_path,
                        shape=(hidden_size,),
                        )

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_final_layer_norm_weight[:] = out.numpy()
            model_decoder_final_layer_norm_weight.flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_final_layer_norm_bias[:] = out.numpy()
            model_decoder_final_layer_norm_bias.flush()
            # If code fails here, you need to update transformers.
            # An earlier version was missing the final_layer_norm


        for layer_i in range(num_hidden_layers):

            # K/V/Q weights
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.self_attn")
            if shard_i == 0:
                model_decoder_layers_self_attn_k_proj_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.k_proj.weight",
                        base_path,
                        shape=(num_attention_heads, dims_per_head, hidden_size),
                        )
                model_decoder_layers_self_attn_v_proj_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.v_proj.weight",
                        base_path,
                        shape=(num_attention_heads, dims_per_head, hidden_size),
                        )
                model_decoder_layers_self_attn_q_proj_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.q_proj.weight",
                        base_path,
                        shape=(num_attention_heads, dims_per_head, hidden_size),
                        )

            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, hidden_size))
            model_decoder_layers_self_attn_k_proj_weight[layer_i][get_slice(heads_per_shard, shard_i), :, :] = out.numpy()
            model_decoder_layers_self_attn_k_proj_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, hidden_size))
            model_decoder_layers_self_attn_v_proj_weight[layer_i][get_slice(heads_per_shard, shard_i), :, :] = out.numpy()
            model_decoder_layers_self_attn_v_proj_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, hidden_size))
            model_decoder_layers_self_attn_q_proj_weight[layer_i][get_slice(heads_per_shard, shard_i), :, :] = out.numpy()
            model_decoder_layers_self_attn_q_proj_weight[layer_i].flush()


            # K/V/Q bias
            if shard_i == 0:
                model_decoder_layers_self_attn_k_proj_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.k_proj.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_layers_self_attn_v_proj_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.v_proj.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_layers_self_attn_q_proj_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.q_proj.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model_decoder_layers_self_attn_k_proj_bias[layer_i][get_slice(hidden_size_per_shard, shard_i)] = out.numpy()
            model_decoder_layers_self_attn_k_proj_bias[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model_decoder_layers_self_attn_v_proj_bias[layer_i][get_slice(hidden_size_per_shard, shard_i)] = out.numpy()
            model_decoder_layers_self_attn_v_proj_bias[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model_decoder_layers_self_attn_q_proj_bias[layer_i][get_slice(hidden_size_per_shard, shard_i)] = out.numpy()
            model_decoder_layers_self_attn_q_proj_bias[layer_i].flush()


            # O weight, O bias
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.self_attn.out_proj")
            if shard_i == 0:
                model_decoder_layers_self_attn_out_proj_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.out_proj.weight",
                        base_path,
                        shape=(hidden_size, num_attention_heads, dims_per_head),
                        )
                model_decoder_layers_self_attn_out_proj_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn.out_proj.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
            out, flat_params = take_out(flat_params, (hidden_size, heads_per_shard, dims_per_head))
            model_decoder_layers_self_attn_out_proj_weight[layer_i][:, get_slice(heads_per_shard, shard_i), :] = out.numpy()
            model_decoder_layers_self_attn_out_proj_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_self_attn_out_proj_bias[layer_i][:] = out.numpy()
            model_decoder_layers_self_attn_out_proj_bias[layer_i].flush()


            # Input LayerNorm
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.self_attn_layer_norm")
            if shard_i == 0:
                model_decoder_layers_self_attn_layer_norm_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn_layer_norm.weight",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_layers_self_attn_layer_norm_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.self_attn_layer_norm.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_self_attn_layer_norm_weight[layer_i][:] = out.numpy()
            model_decoder_layers_self_attn_layer_norm_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_self_attn_layer_norm_bias[layer_i][:] = out.numpy()
            model_decoder_layers_self_attn_layer_norm_bias[layer_i].flush()


            # MLP dense_h_to_4h
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.fc1")
            if shard_i == 0:
                model_decoder_layers_fc1_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.fc1.weight",
                        base_path,
                        shape=(ffn_dim, hidden_size),
                        )
                model_decoder_layers_fc1_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.fc1.bias",
                        base_path,
                        shape=(ffn_dim,),
                        )
            out, flat_params = take_out(flat_params, (ffn_dim_per_shard, hidden_size))
            model_decoder_layers_fc1_weight[layer_i][get_slice(ffn_dim_per_shard, shard_i), :] = out.numpy()
            model_decoder_layers_fc1_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (ffn_dim_per_shard,))
            model_decoder_layers_fc1_bias[layer_i][get_slice(ffn_dim_per_shard, shard_i)] = out.numpy()
            model_decoder_layers_fc1_bias[layer_i].flush()


            # MLP dense_4h_to_h
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.fc2")
            if shard_i == 0:
                model_decoder_layers_fc2_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.fc2.weight",
                        base_path,
                        shape=(hidden_size, ffn_dim),
                        )
                model_decoder_layers_fc2_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.fc2.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
            out, flat_params = take_out(flat_params, (hidden_size, ffn_dim_per_shard))
            model_decoder_layers_fc2_weight[layer_i][:, get_slice(ffn_dim_per_shard, shard_i)] = out.numpy()
            model_decoder_layers_fc2_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_fc2_bias[layer_i][:] = out.numpy()
            model_decoder_layers_fc2_bias[layer_i].flush()


            # Post-attention LayerNorm
            print(shard_i, layer_i, f"model.decoder.layers.{layer_i}.final_layer_norm")
            if shard_i == 0:
                model_decoder_layers_final_layer_norm_weight[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.final_layer_norm.weight",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_layers_final_layer_norm_bias[layer_i] = create_mmap_file(
                        f"model.decoder.layers.{layer_i}.final_layer_norm.bias",
                        base_path,
                        shape=(hidden_size,),
                        )
            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_final_layer_norm_weight[layer_i][:] = out.numpy()
            model_decoder_layers_final_layer_norm_weight[layer_i].flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_layers_final_layer_norm_bias[layer_i][:] = out.numpy()
            model_decoder_layers_final_layer_norm_bias[layer_i].flush()


        if not load_final_layer_norm_first:
            print(shard_i, layer_i, "model.decoder.final_layernorm")
            # Post-attention LayerNorm
            if shard_i == 0:
                model_decoder_final_layernorm_weight = create_mmap_file(
                        "model.decoder.final_layernorm.weight",
                        base_path,
                        shape=(hidden_size,),
                        )
                model_decoder_final_layernorm_bias = create_mmap_file(
                        "model.decoder.final_layernorm.bias",
                        base_path,
                        shape=(hidden_size,),
                        )

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_final_layernorm_weight = out.numpy()
            model_decoder_final_layernorm_weight.flush()

            out, flat_params = take_out(flat_params, (hidden_size,))
            model_decoder_final_layernorm_bias = out.numpy()
            model_decoder_final_layernorm_bias.flush()

            # If code fails here, you need to update transformers.
            # An earlier version was missing the final_layer_norm

        assert flat_params.numel() == 0


load_sharded_weights(base_path, sharded_checkpoint_list)
