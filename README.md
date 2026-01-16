# RP-Gaussian & Style Transfer Collection (PyTorch)

This repository provides a unified framework for artistic style transfer, implementing **4 distinct methods** including AdaAttN, StyleID, and RP-Gaussian features.

## 🚀 Supported Methods

The codebase integrates the following algorithms:

1.  **AdaAttN** (Adaptive Attention Normalization)
2.  **StyleID** (Style Identity)
3.  **RP-Gaussian** (Random Projection Gaussian)
4.  **Other Method** (Please update this with the 4th method name)

## 🛠️ Prerequisites

To run this code, please ensure you have the following dependencies installed:

* Python >= 3.6
* PyTorch >= 1.7
* Torchvision

## 📥 Pre-trained Weights

To run the inference, you need to download the pre-trained weights. Please place the downloaded weights in the corresponding directory (e.g., `checkpoints/`).

| Method | Weights Source / Reference |
| :--- | :--- |
| **AdaAttN** | [https://github.com/Huage001/AdaAttN](https://github.com/Huage001/AdaAttN) |
| **StyleID** | [https://github.com/jiwoogit/StyleID](https://github.com/jiwoogit/StyleID) |
| **RP-Gaussian** | *Weights included in this repo / See release* |
| **Other** | *N/A* |

## 💻 Usage

Use the `test.py` script to generate stylized images. You can specify the input content and style directories.

### Run Command

```bash
python test.py --content input/content --style input/style
