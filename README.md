# Deep Learning HW4: Learning to Drive with Transformers and CNNs

This repository contains the implementation of a **Learning to Drive** model using Transformers and Convolutional Neural Networks (CNNs) as part of Homework 4 for a Deep Learning course. The model is trained on the SuperTuxKart Drive Dataset to predict driving actions based on input visual data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
In this project, we implement a hybrid model combining Transformers and CNNs to learn driving behaviors from visual inputs. The CNN extracts spatial features, while the Transformer captures temporal dependencies in sequential driving frames. This approach demonstrates the effectiveness of deep learning in autonomous driving applications.

## Features
- **Hybrid Architecture:** Combines CNN for feature extraction and Transformer for sequence modeling.
- **Action Prediction:** Predicts driving actions such as steering, acceleration, and braking.
- **Customizable Parameters:** Adjust model architecture, learning rate, and training duration.
- **Evaluation Metrics:** Calculates accuracy and loss for quantitative assessment.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/Deep-Learning-HW4.git
cd Deep-Learning-HW4
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the training script as follows:

```bash
python drive_model.py --train_file <path_to_training_data> --test_file <path_to_test_data> --num_epochs <epochs> --learning_rate <lr> --batch_size <batch_size>
```

#### Command-line Arguments
- `--train_file`: Path to the training dataset (e.g., images and labels).
- `--test_file`: Path to the test dataset.
- `--num_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for optimization.
- `--batch_size`: Batch size for training.

Example:
```bash
python drive_model.py --train_file data/train.csv --test_file data/test.csv --num_epochs 50 --learning_rate 0.001 --batch_size 32
```

## Example Output
The script will output training and test performance metrics, such as loss and accuracy, at each epoch.

Sample output:
```
Epoch 1/50: Loss = 2.13, Accuracy = 58.0%
Epoch 50/50: Loss = 0.65, Accuracy = 92.3%
Final Test Accuracy: 90.8%
```

## Dependencies
This implementation uses the following dependencies:
- `numpy`
- `torch`
- `torchvision`
- `tqdm`

Install the dependencies using:
```bash
pip install numpy torch torchvision tqdm
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore, adapt, and extend the code for your own experiments or projects. Contributions are welcome!
