#!/usr/bin/env python3
"""Extract features with pretrained model from RetCCL: Clustering-guided Contrastive Learning for Whole-slide Image Retrieval

https://github.com/Xiyue-Wang/RetCCL

use like this:
    python -m marugoto.extract.xiyue_wang \
        --checkpoint-path ~/Downloads/best_ckpt.pth \
        --outdir ~/TCGA_features/TCGA-CRC-DX-features/xiyue-wang \
        /mnt/TCGA_BLOCKS/TCGA-CRC-DX-BLOCKS/*
"""
# %%
import hashlib
from pathlib import Path
import timm
import torch
import torch.nn as nn
from marugoto.extract.extract import extract_features_
import uni

# %%


def extract_uni_features_(
    *slide_tile_paths: Path, checkpoint_path: str, **kwargs
):
    """Extracts features from slide tiles.

    Requirements: 
        Permission from authors via huggingface:
            https://huggingface.co/MahmoodLab/UNI
        Huggingface account with valid login token
        Downloaded UNI model

    Args:
        checkpoint_path:  Path to the downloaded model checkpoint file.
    """
    # calculate checksum of model
    sha256 = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    digest = sha256.hexdigest()

    # From UNI repo instructions, with downloaded model
#    model = timm.create_model(
#        "vit_large_patch16_224",
#        img_size=224, patch_size=16,
#        init_values=1e-5, num_classes=0, dynamic_img_size=True
#        )
    
    model, transform = uni.get_encoder(
        enc_name="uni", device='cpu', assets_dir='/mnt/c/Renal-SMLM/uni-model'
        )
    
#    uni = torch.load(checkpoint_path,
#                     map_location='cpu',
#                     weights_only=True
#                     )
# model.load_state_dict(uni, strict=True)

    # And UNI repo instructions say create a transform for normalisation
    # after this (but we do this later anyway)

    # Try this, as for ctranspath
    if torch.cuda.is_available():
        model = model.to('cuda')  # cuda:0 ??

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model.cuda(),
        model_name=f"mahmood-uni-{digest[:8]}",
        **kwargs,
    )
