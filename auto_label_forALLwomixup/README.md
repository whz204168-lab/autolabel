# auto-label for all the signal 


## Requirements
- Python 3.6+
- PyTorch 1.0
- **torchvision 0.2.2 (older versions are not compatible with this code)** 
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train
Train the model by 600 labeled data of RML2016 dataset:

auto-label for all the dataset.

without mixup


```
python train.py
```


### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10@250
```

## Results (Accuracy)



## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```