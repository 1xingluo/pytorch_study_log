# -*- coding: utf-8 -*-
"""
兼容 Fairseq 0.10.x 的 Transformer 模型
直接用 build_transformer_model(default_model_args, src_dict, tgt_dict) 构建
"""

import torch
from types import SimpleNamespace
from fairseq.models.transformer import TransformerModel

# =============================
# 默认模型参数
# =============================
default_model_args = SimpleNamespace(
    encoder_embed_dim=256,
    encoder_layers=1,
    encoder_attention_heads=4,
    encoder_ffn_embed_dim=2048,
    encoder_normalize_before=False,
    decoder_embed_dim=256,
    decoder_layers=1,
    decoder_attention_heads=4,
    decoder_ffn_embed_dim=2048,
    decoder_normalize_before=False,
    share_decoder_input_output_embed=False,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    max_source_positions=1024,
    max_target_positions=1024,
    activation_fn='relu',
)


# =============================
# 构建 Transformer 模型
# =============================
def build_transformer_model(model_args, src_dict, tgt_dict):
    if src_dict is None or tgt_dict is None:
        raise ValueError("❌ build_transformer_model() 需要传入 source/target 词典！")

    # -----------------------------
    # 为 Fairseq 0.10.x 补全所有必要属性
    # -----------------------------
    args = SimpleNamespace(
        # encoder / decoder 维度
        encoder_embed_dim=model_args.encoder_embed_dim,
        encoder_ffn_embed_dim=model_args.encoder_ffn_embed_dim,
        encoder_layers=model_args.encoder_layers,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_embed_dim=model_args.decoder_embed_dim,
        decoder_ffn_embed_dim=model_args.decoder_ffn_embed_dim,
        decoder_layers=model_args.decoder_layers,
        decoder_attention_heads=model_args.decoder_attention_heads,
        share_decoder_input_output_embed=model_args.share_decoder_input_output_embed,
        dropout=model_args.dropout,
        attention_dropout=model_args.attention_dropout,
        activation_dropout=model_args.activation_dropout,
        activation_fn=model_args.activation_fn,
        encoder_normalize_before=model_args.encoder_normalize_before,
        decoder_normalize_before=model_args.decoder_normalize_before,
        max_source_positions=model_args.max_source_positions,
        max_target_positions=model_args.max_target_positions,

        # 新增缺省值，防止 AttributeError
        encoder_layers_to_keep=None,
        decoder_layers_to_keep=None,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        encoder_embed_path=None,
        decoder_embed_path=None,
        no_token_positional_embeddings=False,
        adaptive_input=None,
        adaptive_input_factor=1.0,
        adaptive_input_dropout=0.0,
        quant_noise_pq=0.0,
        quant_noise_pq_block_size=8,
    )

    model = TransformerModel.build_model(
        args=args,
        task=SimpleNamespace(
            source_dictionary=src_dict,
            target_dictionary=tgt_dict
        )
    )
    return model
