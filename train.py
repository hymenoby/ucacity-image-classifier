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
parser = argparse.ArgumentParser(description="Training neural network")
parser.add_argument('data_directory', action='store',
                    help='Folder of the datasets must have train, test and valid folders')
parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='The pretrained architecture to use, mus be part of torchvision.models, default vgg11')
parser.add_argument('--hidden_units', action='store', default=[2000],
                    dest='hidden_layers', nargs="+", type=int,
                    help='The number of hidden units in classifier, space separated numbers, default 2000')
parser.add_argument('--dropout', action='store', default=0.5,
                    dest='dropout', type=float,
                    help='The dropout of the classifier, default 0.5')
parser.add_argument('--lr', action='store', default=0.0001,
                    dest='learning_rate', type=float,
                    help='The learning rate of the network, default 0.0001')
parser.add_argument('--epoch', action='store', default=1,
                    dest='epoch', type=int,
                    help='The number of epochs, default 1')
parser.add_argument('--print_every', action='store', default=32,
                    dest='print_every', type=int,
                    help='The number of steps before printing the progress, default 32')
parser.add_argument('--gpu', action="store_true", default=False,
                    dest="use_gpu",
                    help='Use the gpu or not, default off')
parser.add_argument('--save_dir', action="store", default="./",
                    dest="save_dir", type=str,
                    help='The path of the directory to save the checkpoint to')
parser.add_argument('--resume_training', action="store_true", default=False,
                    dest="resume_training",
                    help='Resume some saved training, default off')
arguments = parser.parse_args()

# check the arguments
if not os.path.exists(arguments.data_directory):
    print("THE DATA FOLDER <<" + arguments.data_directory + ">> DOES NOT EXISTS")
    sys.exit(errno.EINVAL)

model_name = arguments.pretrained_model
if not (model_name.islower() and not model_name.startswith("__") and hasattr(models, model_name)):
    print("THE MODEL <<" + arguments.pretrained_model + ">> DOES NOT EXISTS IN THE <<torchvision.models>> MODULE")
    sys.exit(errno.EINVAL)

device = torch.device('cpu')
if arguments.use_gpu:
    # get the available device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("THE GPU IS NOT AVAILABLE")
        sys.exit(errno.EINVAL)

SAVE_FILE_NAME = "checkpoint.pth"
SAVE_FILE_PATH = arguments.save_dir + SAVE_FILE_NAME

if arguments.save_dir:
    if not os.path.exists(arguments.save_dir):
        try:
            os.mkdir(arguments.save_dir)
        except OSError:
            print("Creation of the directory %s failed" % arguments.save_dir)

if arguments.resume_training:
    if not arguments.save_dir:
        print("YOU MUST SPECIFY THE SAVE DIRECTORY (--save_dir) IF YOU WANT TO RESUME A TRAINING")
        sys.exit(errno.EINVAL)
    else:
        if not os.path.exists(SAVE_FILE_PATH):
            print("THERE IS NOT CHECKPOINT IN THE <<" + arguments.save_dir + ">> FOLDER")
            sys.exit(errno.EINVAL)

# print the parameters before starting action
print(
    "- - - - - - - - - - TRAINING NETWORK - - - - - - - - - -\n",
    "Data directory: {}\n"
    "Architecture: {}\n"
    "Hidden units: {}\n"
    "Dropout: {}\n"
    "Epochs: {}\n"
    "Print every: {}\n"
    "Use GPU: {}\n"
    "Save dir: {}\n"
    "Resume training: {}\n"
        .format(arguments.data_directory, arguments.pretrained_model, arguments.hidden_layers, arguments.dropout,
                arguments.epoch, arguments.print_every, arguments.use_gpu, arguments.save_dir,
                arguments.resume_training)
)

answer = input("CONTINUE WITH THESE PARAMETERS [Y/N]? ").lower()

if answer == 'n':
    sys.exit()
print("\n\n")

# get the dataloaders and the datasets
dataloaders, image_datasets = utils.get_images_datasets_and_dataloaders(arguments.data_directory)

# Build the network
if not arguments.resume_training:
    print("- - - - Building the network . . . ")
    # if the training is not resumed build a new network
    model, layers = utils.build_model(arguments.pretrained_model, image_datasets["train"],
                                      arguments.hidden_layers, arguments.dropout)
    optimizer = optim.Adam(getattr(model, layers["layer_name"]).parameters(), arguments.learning_rate)
else:
    print("- - - - Loading the network . . . ")
    # if the training is resumed load a network from a checkpoint
    model, optimizer, layers = utils.load_model(SAVE_FILE_PATH, device)
print("\n\n")

# set the criterion
criterion = nn.NLLLoss()

# Train the network
# Monitor the training time of the whole model
print("- - - - Training the network . . . ")
start_time = time.time()
# train the network
utils.train_network(model, dataloaders['train'], optimizer, criterion, arguments.epoch,
                    dataloaders['validation'], device, arguments.print_every)
print("- - - - Training time {}".format(time.time() - start_time), "\n\n")

# test the network
print("- - - - Testing the network . . . ")

model.eval()
# disable grad
with torch.no_grad():
    accuracy, test_loss = utils.validate(model, dataloaders["test"], criterion, device)
# print status
print("Test Loss: {:.4f}.. ".format(test_loss / len(dataloaders["test"])),
      "Test Accuracy: {:.4f}".format(accuracy / len(dataloaders["test"])), "\n\n")

# save the trained model
print("- - - - Saving the model . . . ")
utils.save_model(model, arguments.pretrained_model, layers, image_datasets["train"],
                 optimizer, SAVE_FILE_PATH)
print("Model saved!", "\n\n")
