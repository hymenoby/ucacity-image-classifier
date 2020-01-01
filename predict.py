import argparse
import errno
import sys
import time
import torch
import os
import utils
from torchvision import models
from torch import nn
from torch import optim

# get arguments
parser = argparse.ArgumentParser(description="Predict image")
parser.add_argument('image_path', action='store',
                    help='Path of the image to test')
parser.add_argument('checkpoint', action='store',
                    help='Path of the model checkpoint file')
parser.add_argument('--gpu', action="store_true", default=False,
                    dest="use_gpu",
                    help='Use the gpu or not, default off')
parser.add_argument('--category_names', action="store", default="./cat_to_name.json",
                    dest="category_map_file", type=str,
                    help='The path of the json file mapping classes to names')
parser.add_argument('--topk', action="store", default=1,
                    dest="topk", type=int,
                    help='The number of predicted classes to get')
arguments = parser.parse_args()

# check the arguments

if arguments.image_path:
    if not os.path.exists(arguments.image_path):
        print("THE IMAGE FILE <<"+arguments.image_path+">> DOES NOT EXIST")
        sys.exit(errno.EINVAL)

if arguments.checkpoint:
    if not os.path.exists(arguments.checkpoint):
        print("THE CHECKPOINT FILE <<"+arguments.checkpoint+">> DOES NOT EXIST")
        sys.exit(errno.EINVAL)

if arguments.category_map_file:
    if not os.path.exists(arguments.category_map_file):
        print("THE CLASS MAP FILE <<"+arguments.category_map_file+">> DOES NOT EXIST")
        sys.exit(errno.EINVAL)

device = torch.device('cpu')
if arguments.use_gpu:
    # get the available device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("THE GPU IS NOT AVAILABLE")
        sys.exit(errno.EINVAL)

# get class to name map from file
class_name_map = utils.get_cat_to_name_map(arguments.category_map_file)

# get the predictions
probs, classes, names = utils.predict(arguments.image_path, arguments.checkpoint,
                                      class_name_map, device, arguments.topk)

predictions = list(zip(classes, names, probs))

print("\n")
print("- - - - Prediction - - - -", "\n")

print("Top {} classes".format(arguments.topk))
for prediction in predictions:
    print("\t Class: {:4}, Name: {:20}, Probability: {:.4f}".format(*prediction))

print("\n")
print("The picture has a << {:.2f}% >> probability to be a << {} >> ".format(predictions[0][2]*100, predictions[0][1]))
print("\n\n")
