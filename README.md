# Swin Transformer Backbone for Full-Reference IQA

An exploratory experiment that replaces the CNN backbone in transformer-based image quality assessment with a **Swin Transformer**. It was part of our work for the NTIRE 2022 Perceptual Image Quality Assessment Challenge on the [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) dataset.

The final method and paper are in **[IQA-multiscaling](https://github.com/AbhisekKeshari/IQA-multiscaling)** ([arXiv:2204.09779](https://arxiv.org/abs/2204.09779)).

## Idea

IQT-style models use a convolutional backbone (InceptionResNetV2). This experiment tests whether a hierarchical vision transformer gives better perceptual features for full-reference IQA:

- **Backbone:** `swin_large_patch4_window7_224_in22k` (ImageNet-22k pretrained, via [timm](https://github.com/huggingface/pytorch-image-models)), fine-tuned end to end
- **Features:** Final-stage (stage 4) token features of the reference and distorted 224×224 crops, captured with a forward hook
- **Head:** Concatenated reference and distorted features, average pooling, then an MLP (1536 → 1024 → 256 → 1) that predicts the quality score
- **Training:** MSE loss, Adam (lr 1e-6) with cosine annealing, 9:1 scene-level split; the checkpoint with the best SROCC + PLCC is kept

## Structure

```
model_main.py        # config + training entry point
model/swin_class.py  # Swin backbone + regression head
model/trainer.py     # train / eval loops (SROCC, PLCC)
util.py              # augmentations and data split
option/config.py     # config helper
```

## Notes

Research code from 2022, kept for reference. The PIPAL data loader (`dataset/data_PIPAL.py`) is not in this repo; the loader in [IQA-multiscaling](https://github.com/AbhisekKeshari/IQA-multiscaling/tree/main/IQT-multiscale%20submission/data) can be used. Requires `torch`, `timm`, `scipy` and `opencv-python`.
