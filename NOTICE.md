# DINO-SFF attribution

This package contains the C10 DINOv3 **ConvNeXt Base** model, dataset loaders and transforms derived from the local GLF-Net project, evaluation utilities and experiment-specific decoder modules. The active files are under `geoseg/`; the historical `vendor/` and other experiment variants mentioned below are not distributed in this package.

The backbone initialization used here is `timm/convnext_base.dinov3_lvd1689m`, with its exact revision and checksum in `pretrained_manifest.json`. The upstream DINOv3 terms apply to pretrained model materials. The original GLF-Net README declares CC BY-NC-SA 4.0; retain upstream attribution and applicable terms. The following historical provenance is preserved rather than replaced.

# Source attribution

`vendor/vHeat.py`, `vendor/vaihingen_dataset.py`, and `vendor/transform.py`
were copied from the user's GLF-Net checkout for a self-contained experiment.
The original repository README declares CC BY-NC-SA 4.0 for its materials.
Retain original attribution and applicable noncommercial/share-alike terms
when redistributing this derivative suite.

Original repository: https://github.com/tangyp8321/GLF-Net

Backbone research: https://arxiv.org/abs/2405.16555

The ten decoder implementations are experiment-specific combinations inspired
by the research cited in RESEARCH.md, not official reproductions of those models.

Round six reuses the S09 decoder from the user's previous experiments. The
Q17 conditional-prototype modules used in round four are not included in these
ten variants. Backbone adapters and adaptive heat/readout modifications are
experiment-specific implementations inspired by the papers in RESEARCH.md,
not official implementations of ConvNeXt V2, FNO, Restormer, NAFNet or RAFT.

DINOv3 initialization uses the public timm maintainer release
https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m under its DINOv3
license. The downloader retains the model card, license, configuration and
pinned revision provenance alongside the verified weights. timm provides
the ConvNeXt implementation; no gated Meta endpoint is bypassed.

A copy of the upstream DINOv3 agreement is included at `third_party/DINOv3_LICENSE.md`. It was retrieved from the official facebookresearch/dinov3 repository.
