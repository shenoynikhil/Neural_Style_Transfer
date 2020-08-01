import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os
import sys
import argparse
import datetime

parser = argparse.ArgumentParser(description='Neural Algorithm of Artistic Style')
parser.add_argument('--content-path', type=str, required=True, help='path to content image')
parser.add_argument('--style-path', type=str, required=True, help='path to style image')
parser.add_argument('--output', type=str, default="examples/", required=False, help='output directory path')
parser.add_argument('--content-layers', type=int, nargs='+', default=[4], required=False, help='list of numbers indicating convolutional layers to use for content activations')
parser.add_argument('--style-layers', type=int, nargs='+', default=[1,2,3,4,5], required=False, help='list of numbers indicating convolutional layers to use for style activations')
parser.add_argument('--content-weight', type=float, default=1.0, required=False, help='factor controlling contribution of content loss to total loss')
parser.add_argument('--style-weight', type=float, default=100.0, required=False, help='factor controlling contribution of style loss to total loss')
parser.add_argument('--learning-rate', type=float, default=1.0, required=False)
parser.add_argument('--num-steps',type=int, default=400, required=False, help="number of steps to be taken by optimizer")
parser.add_argument('--image-size', type=int, default=224, required=False, help="dimension of image")
parser.add_argument('--use-gpu', action='store_true' , help="use gpu for computation")


# Some Utility Functions
def compute_content_loss(content_activations, input_activations):
    loss = 0
    for content_act, input_act in zip(content_activations, input_activations):
        loss += F.mse_loss(content_act, input_act)
    return loss

def compute_style_loss(content_activations, input_activations):
    loss = 0
    for content_act, input_act in zip(content_activations, input_activations):
        loss += F.mse_loss(content_act, input_act)
    return loss

def gram_matrix(x):
    return torch.matmul(x, x.t()) / (x.shape[0] ** 2)

class StyleGenerator(nn.Module):
    def __init__(self, style_path, content_path, style_layers = [1, 2, 3, 4, 5], content_layers = [4]):
        '''
        Inputs:
            style_path: Path to style image
            content_path: Path to content image
            style_layers: nn.Conv2D layers that are required for style activations
            content_layers: nn.Conv2D layers that are required for content activations
        '''
        super(StyleGenerator, self).__init__()

        # Set image paths
        self.style_path = style_path
        self.content_path = content_path

        # Set required conv layers for content and style activations
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # Initialize vgg model and all params are not learnable
        self.net = models.vgg19(pretrained = True).features.to(device)
        for layer in self.net:
            layer.requires_grad = False
        
        # Set preprocessing transforms based on vgg type inputs
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
        self.postprocess = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                               ])

    def getActivations(self, x):
        '''
        Input:
            x: An Image tensor of the form (n, B, H, W)
        Output: 
            content_activations: List of activated flatenned activated tensors
            style_activations: List of activated style tensors (shape : (C, H* W))
        '''
        conv_count = 0
        max_count = max(max(self.content_layers), max(self.style_layers))
        
        # stores indices from the vgg network of the Conv2D layers
        style_activations = []
        content_activations = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                if conv_count in self.style_layers:
                    style_activations.append(x.view(x.shape[1], -1))
                if conv_count in self.content_layers:
                    content_activations.append(x.flatten())
        return style_activations, content_activations

    def generate_stylized_image(self, output_dir = "/content", content_wt = 1.0, style_wt = 100.0, 
                                learning_rate = 1.0, num_steps = 500, image_size = (224, 224)):
        '''
        Inputs:
            content_path: path to content image
            style_path: path to style image
            output_dir: output directory for intermediate generated images
            
            content_wt: factor controlling contribution of content loss to total loss
            style_wt: factor controlling contribution of style loss to total loss
            
            learning_rate: learning rate of optimizer
            num_steps: number of steps to be taken by optimizer
            
            image_size: tuple indicating size of input and output images to be used, default is input size expected by VGG19
        '''
        # Create output directory
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Load content and style images and compute relevant activations
        content_img = Image.open(self.content_path).resize(image_size)
        style_img = Image.open(self.style_path).resize(image_size)
        
        content = self.preprocess(content_img).unsqueeze(0).to(device)
        style = self.preprocess(style_img).unsqueeze(0).to(device)

        # get content and style activations from getActivation and they are not learnable
        content_activations = [x.detach() for x in self.getActivations(content)[1]]
        style_activations = [gram_matrix(x.detach()) for x in self.getActivations(style)[0]]

        # initialize output image as content_image + gaussian_noise for faster convergence
        gaussian_noise = torch.clamp(torch.randn(1, 3, image_size[0], image_size[1]), -1, 1).to(device)
        gen_image = content * 0.5 + gaussian_noise * 0.5
        gen_image = nn.Parameter(gen_image)

        # initialize optimizer with gen_image as parameters over which optimization is carried out
        optimizer = torch.optim.LBFGS([gen_image.requires_grad_()], lr=learning_rate)

        steps = [0]
        while steps[0]<num_steps:
            def closure():
                '''
                closure function required by LBFGS optimizer
                '''
                optimizer.zero_grad()
                
                inp_style, inp_content = self.getActivations(gen_image)
                inp_style = [gram_matrix(x) for x in inp_style]

                content_loss = compute_content_loss(content_activations, inp_content)
                style_loss = compute_style_loss(style_activations, inp_style)
                loss = content_wt * content_loss + style_wt * style_loss
                steps[0] += 1
                if steps[0] % 50 == 0:
                    print("Num Steps: {} \tContent Loss: {} \tStyle Loss: {} \tTotal Loss:{}".format(steps[0], 
                                round(content_loss.item(),3), round(style_loss.item(),3), round(loss.item(),3)))

                loss.backward()
                return loss
            optimizer.step(closure)

        fig,ax = plt.subplots(1,3, figsize=(15,7))
        ax[0].imshow(content_img)
        ax[0].set_title("Content")
        ax[1].imshow(style_img)
        ax[1].set_title("Style")
        ax[2].imshow(torch.clamp(self.postprocess(gen_image[0].cpu().detach()).permute(1, 2, 0),0, 1).numpy())
        ax[2].set_title("Generated Image")
        plt.show()
        fig.savefig(os.path.join(output_dir,"final.jpg"))

if __name__=="__main__": 
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: {}".format(device))
    styler = StyleGenerator(style_path=args.style_path, content_path=args.content_path, style_layers=args.style_layers, content_layers=args.content_layers)
    styler.generate_stylized_image(output_dir=args.output, 
                            content_wt=args.content_weight, style_wt=args.style_weight, 
                            learning_rate=args.learning_rate, num_steps=args.num_steps,
                            image_size=(args.image_size,args.image_size))

