# Neural Network From Scratch

Most of the code from [Make Your Own Neural Network](https://www.amazon.in/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G) by [Tariq Rashid](https://github.com/makeyourownneuralnetwork)

## Requirements

- numpy
- scipy
- [mnist dataset](https://pjreddie.com/projects/mnist-in-csv/)

## Usage

`python main.py` for hardcoded values

`python main.py 1 HiddenNodes LearningRate Epochs` for testing optimum performance

`python main.py 2 HiddenNodes LearningRate Epochs` for custom values

## Benchmark

| Hidden Nodes | Learning Rate | Epochs | Accuracy |
| ------------ | ------------- | ------ | -------- |
| 175          | 0.2           | 7      | 97.04%   |
| 175          | 0.2           | 5      | 96.99%   |
| 250          | 0.2           | 5      | 96.90%   |
| 200          | 0.2           | 5      | 96.89%   |
| 250          | 0.2           | 7      | 96.89%   |
