# Project Title: PGPM

## Overview
The PGPM project is designed for signal segmentation using a U-Net architecture. It includes functionalities for generating synthetic data, training the model, and running inference to obtain predictions.

## Directory Structure
```
pgpm
├── src
│   ├── datagenerator.py          # Functions for generating synthetic data and datasets
│   ├── model
│   │   ├── __init__.py           # Initializes the model module
│   │   └── unet1sc.py            # Defines the UNet1SC class for segmentation tasks
│   ├── training
│   │   ├── __init__.py           # Initializes the training module
│   │   └── train_utils.py         # Contains train_epoch and val_epoch functions
│   ├── train.py                   # Main entry point for training the model
│   └── inference.py               # Runs inference on the trained model
├── Parameters
│   └── params.json                # Configuration parameters for training and data generation
├── savedmodels                    # Directory for storing saved model checkpoints
├── Figures                        # Directory for storing generated figures and plots
├── requirements.txt               # Lists project dependencies
└── README.md                      # Documentation and instructions for the project
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Generation**: Use the `datagenerator.py` to create synthetic datasets for training and validation.
2. **Training**: Run `train.py` to start the training process. This will initialize the model, set up data loaders, and call the training functions.
3. **Inference**: Use `inference.py` to run inference on the trained model and generate predictions.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.