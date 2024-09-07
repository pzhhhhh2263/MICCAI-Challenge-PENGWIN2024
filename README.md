# PENGWIN2024

Source code for [MICCAI2024 Challenge-PENGWIN](https://pengwin.grand-challenge.org/)

## Project Overview
This repository contains the code for the PENGWIN 2024 challenge, which focuses on medical image segmentation. The challenge involves segmenting pelvic bone fragments and is part of the MICCAI 2024 competition. The models and scripts provided here were used for training and evaluating segmentation models as part of the competition submission.


## Development Environment
- **OS**: Ubuntu 20.04
- **GPU**: NVIDIA RTX 3090
- **CUDA**: 11.8
- **Python**: 3.9
- **Pytorch**: 2.3.1
- **Torchvision**: 0.18.1
  
## Requirements
```
pip install -r requirements.txt
```



## TASK1

### Training

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -tr nnUNetTrainer_NexToU_ep500_CosAnneal

```




## Acknowledgment
Big thanks to the following repositories for their contributions and support:
- [NexToU](https://github.com/PengchengShi1220/NexToU)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

