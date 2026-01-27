## Multi-task Regression with Deep Learning: Digital Typhoon Dataset

This repository implements Deep Learning techniques for **Multi-task Regression**, specifically focused on estimating tropical cyclone intensity (Wind Speed and Central Pressure). We compare the performance of traditional Residual Networks (**ResNet50**) against **Vision Transformers (ViT)** using the long-term Digital Typhoon satellite dataset.

### Dataset Overview

The project utilizes the **Digital Typhoon** dataset, a comprehensive spatio-temporal collection of satellite imagery. The project samples 15% of teh typhoons.

* **Source:** Infrared (IR) observations from Himawari satellites (1978–present).
* **Format:**  pixel images at ~5km resolution.
* **Scale:** 1,099 typhoons and 189,364 images.
* **Targets:** Simultaneous prediction of **Wind Speed (kt)** and **Central Pressure (hPa)**.

### Repository Structure

| File | Description |
| --- | --- |
| `multitask_regression.ipynb` | Initial implementation using **ResNet50** as the backbone. |
| `multitask_regression_v2.ipynb` | **Enhanced version.** Includes the testing of **Vision Transformer (ViT)**. |
| `sample_v2.py` | Strategy script for sampling fractions of the Digital Typhoon dataset for efficient training. |

### Experiments & Results

We conducted multiple experiments to evaluate the impact of backbone freezing and architecture choice on multi-task regression.

### 1. ResNet50 Configuration

ResNet50 served as our baseline. We found that unfreezing the backbone significantly improved the model's ability to extract typhoon-specific features.

### 2. Vision Transformer (ViT) Configuration

Implemented in the `_v2` notebook, ViT uses self-attention to capture global spatial dependencies. In our tests, it required more tuning to compete with the CNN baseline.

### Comparative Performance

| Model | Backbone State | Backbone LR | Head LR | Wind MAE (kt) | Pressure MAE (hPa) |
| --- | --- | --- | --- | --- | --- |
| **ResNet50** | Frozen | - | 1e-3 | 10.04 | 10.11 |
| **ResNet50** | **Unfrozen** | **5e-5** | **1e-4** | **7.86** | **8.03** |
| **ViT** | Base | Frozen | 1e-3 | 15.45 | 16.32 |

> **Conclusion:** The **Unfrozen ResNet50** outperformed ViT for this task, suggesting that the local inductive biases of CNNs are highly effective for the structural patterns found in infrared typhoon imagery at this dataset scale.

## References

1. **Kitamoto, A., et al. (2024).** *Machine Learning for the Digital Typhoon Dataset: Extensions to Multiple Basins...* [arXiv:2411.16421](https://doi.org/10.48550/arXiv.2411.16421)
2. **Kitamoto, A., et al. (2023).** *Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling...* [arXiv:2311.02665](https://doi.org/10.48550/arXiv.2311.02665)
