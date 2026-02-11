## Multi-task Regression with Deep Learning: Digital Typhoon Dataset

This repository implements Deep Learning techniques for **Multi-task Regression**, specifically focused on estimating tropical cyclone intensity (Wind Speed and Central Pressure). We compare the performance of traditional Residual Networks (**ResNet50**) against **Vision Transformers (ViT)** using the long-term Digital Typhoon satellite dataset.

### Dataset Overview

The project utilizes the **Digital Typhoon** dataset, a comprehensive spatio-temporal collection of satellite imagery. The project samples 15% of the typhoons for efficient training.

* **Source:** Infrared (IR) observations from Himawari satellites (1978–present).
* **Format:** Pixel images at ~5km resolution.
* **Scale:** 1,099 typhoons and 189,364 images (sampled subset used for training).
* **Targets:** Simultaneous prediction of **Wind Speed (kt)** and **Central Pressure (hPa)**.

### Repository Structure

```
multitask_regression_deeplearning/
│
├── multitask-regression_final.ipynb   # Initial ResNet50 implementation for multi-task regression
├── multitask-regression_final_v2.ipynb  # Final version with Vision Transformer experiments
├── sample_v2.py  #Data sampling utility for Digital Typhoon dataset to work locally
└── README.md
    
```

### Experiments & Results
I conducted multiple experiments to evaluate the impact of backbone freezing and architecture choice on multi-task regression.  

1. ResNet50 Configuration  
ResNet50 served as the baseline. Unfreezing the backbone significantly improved the model's ability to extract typhoon-specific features from infrared satellite imagery.  
**Key Findings:**
- **Frozen backbone:** Limited feature adaptation, MAE ~10 for both targets
- **Unfrozen backbone:** Better feature extraction with differential learning rates (5e-5 for backbone, 1e-4 for head)
- **Performance improvement:** 21.7% reduction in Wind MAE, 20.6% reduction in Pressure MAE

 2. Vision Transformer (ViT) Configuration  
Implemented in the `_v2` notebook, ViT uses self-attention mechanisms to capture global spatial dependencies across the entire image, unlike CNNs which rely on local convolutions.  
**Key Findings:**
- ViT struggled with the dataset scale and required more extensive tuning
- Global attention patterns may be less effective for localized typhoon features
- Higher MAE compared to ResNet50 baseline (15.45 kt vs 7.86 kt for wind)

#### Comparative Performance  
| Model | Backbone State | Backbone LR | Head LR | Wind MAE (kt) | Pressure MAE (hPa) |
| --- | --- | --- | --- | --- | --- |
| **ResNet50** | Frozen | - | 1e-3 | 10.04 | 10.11 |
| **ResNet50** | **Unfrozen** | **5e-5** | **1e-4** | **7.86** | **8.03** |
| **ViT** | Frozen | - | 1e-3 | 15.45 | 16.32 |

**Conclusion:** The **Unfrozen ResNet50** outperformed ViT for this task, suggesting that the local inductive biases of CNNs are highly effective for the structural patterns found in infrared typhoon imagery at this dataset scale. The hierarchical feature extraction of CNNs captures spiral patterns and eye structures more effectively than global self-attention at the current dataset size.


### Key Insights  
**Architecture Choice Matters:** CNNs excel at capturing local patterns like spiral bands and typhoon eye structures, which are critical for intensity estimation.  
**Backbone Fine-tuning:** Unfreezing the backbone with a lower learning rate allows the model to adapt pre-trained features to domain-specific satellite imagery patterns.  
**Dataset Scale:** The current dataset scale (15% sample) favors CNNs over Transformers, which typically require larger datasets to learn effective attention patterns.  
**Learning Rate Strategy:** Differential learning rates (lower for backbone, higher for head) prevent catastrophic forgetting while allowing task-specific adaptation.  

## References

1. **Kitamoto, A., et al. (2024).** *Machine Learning for the Digital Typhoon Dataset: Extensions to Multiple Basins...* [arXiv:2411.16421](https://doi.org/10.48550/arXiv.2411.16421)
2. **Kitamoto, A., et al. (2023).** *Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling...* [arXiv:2311.02665](https://doi.org/10.48550/arXiv.2311.02665)
