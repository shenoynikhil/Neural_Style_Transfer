# Style Transfer
Pytorch Implementation of Neural Style Transfer based on the paper, <a href = "https://arxiv.org/pdf/1508.06576.pdf">A Neural Algorithm of Artistic Style</a>.

<img src = "./images/final.jpg"/>

### Requirements
* torch
* torchvision
* PIL

### Usage
```
python NeuralStyleTransfer.py --content-path [path to content image] --style-path [path to style image]
```

Run `python NeuralStyleTransfer.py -h` for optional arguments

### References
1. Neural Style Transfer Pytorch Tutorial https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#loss-functions
2. Style Transfer https://github.com/djin31/StyleTransfer
