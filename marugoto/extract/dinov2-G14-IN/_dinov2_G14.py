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
import torch
import torch.nn as nn
from marugoto.extract.extract import extract_features_

# %%


def extract_dino_features_(
    *slide_tile_paths: Path, checkpoint_path: str, **kwargs
):
    """Extracts features from slide tiles.

    Args:
        checkpoint_path (NOT USED!):
            Path to the downloaded model checkpoint file.
    """
    # From dino repo instructions
    model_backbone_name = 'dinov2_vitg14'
    model = torch.hub.load(repo_or_dir='facebookresearch/dinov2',
                           model=model_backbone_name
                           )

    # Try this, as for ctranspath
    if torch.cuda.is_available():
        model = model.to('cuda')  # cuda:0 ??

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model.cuda(),
        model_name=model_backbone_name,
        **kwargs,
    )
