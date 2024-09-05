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

from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem

# %%


def extract_ctranspath_features_(
    *slide_tile_paths: Path, checkpoint_path: str, **kwargs
):
    """Extracts features from slide tiles.

    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX>.
    """
    # calculate checksum of model
    sha256 = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert (
        sha256.hexdigest()
        == "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"
    )

#    model = ResNet.resnet50(
#        num_classes=128, mlp=False, two_branch=False, normlinear=True
#    )
#    pretext_model = torch.load(checkpoint_path)
#    model.fc = nn.Identity()
#    model.load_state_dict(pretext_model, strict=True)

    model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()

    ctranspath = torch.load(checkpoint_path,
                            map_location=torch.device('cpu'),
                            weights_only=True
                            )
    model.load_state_dict(ctranspath['model'], strict=True)

    if torch.cuda.is_available():
        model = model.to('cuda')  # cuda:0 ??

    # STAMP also has FeatureExtractorCTP.transform

    return extract_features_(
        slide_tile_paths=slide_tile_paths,
        model=model.cuda(),
        model_name="xiyuewang-ctranspath-7c998680",
        **kwargs,
    )
