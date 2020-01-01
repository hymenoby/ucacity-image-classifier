import os
import time
import torch
import json
import numpy as np
from classifier_network import ClassifierNetwork
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from PIL import Image


def get_images_datasets_and_dataloaders(data_directory):
    """Get the datasets and dataloaders from the data dir :
        the dir must follow the torch data format

        Args:
            data_directory (str): The model use for validation
        Returns:
            (dataloaders, image_datasets) (tuple of dict) : tuple of dicts containing the train,
                test and validation dataloader
                format:
                    dataloaders = {"train": dataloader, "test: dataloader, "validation": dataloader}
                    image_datasets = {"train": dataset, "test: dataset, "validation": dataset}
        """
    # load dataset and dataloaders
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = {
        "train": transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize]),
        "validation": transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=32),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32)
    }

    return dataloaders, image_datasets


def get_cat_to_name_map(file_path):
    """Get the classes to name mapping of the dataset
        Args:
            file_path (:nn.Module): The model use for validation
        Returns:
            cat_to_name (dict of str): class to name dict mapping
        """
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def build_model(model, train_dataset, hidden_layers, dropout):
    """Function to train a model

    Args:
        model (str): The pretrained model
        train_dataset (:Dataset): The training dataset
        hidden_layers (list of int): The hidden layers list
        dropout (float): The dropout of the classifier
    """
    # get the pretrained network
    model = getattr(models, model)(pretrained=True)

    # freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False

    # get the input layer of our network
    last_layer_name, last_layer_layer = list(model.named_children())[-1]

    input_size = 0
    if isinstance(last_layer_layer, nn.Linear):
        input_size = last_layer_layer.in_features
    elif isinstance(last_layer_layer, nn.Sequential):
        for layer in last_layer_layer:
            if isinstance(layer, nn.Linear):
                input_size = layer.in_features
                break
    else:
        raise Exception("COULD NOT DETERMINE THE NETWORK LAST LAYER INPUT")

    if input_size == 0:
        raise Exception("COULD NOT DETERMINE THE NETWORK LAST LAYER INPUT")

    output_size = len(train_dataset.class_to_idx)

    # create the classifier network and assign it to the last layer of the pretrained model
    classifier = ClassifierNetwork(input_size, output_size, hidden_layers, dropout)
    setattr(model, last_layer_name, classifier)

    layers = {
        "layer_name": last_layer_name,
        "input_size": input_size,
        "output_size": output_size,
        "hidden_layers": hidden_layers
    }

    return model, layers


def train_network(model, train_dataloader, optimizer, criterion, epochs=1,
                  validatation_dataloader=None, device=None, print_every=50):
    """Function to train a model

    Args:
        model (:nn.Module): The model to train
        train_dataloader (:DataLoader): The training dataloader
        validatation_dataloader (:DataLoader, optional): The validation dataloader
        device (:device): The device to use
        optimizer : The optimiser
        criterion : The loss function
        epochs (int, optional) : The number of epoch to run
        print_every (int, optional) : The number of iteration before printing the status

    """
    # if the device is not set, get the device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    steps = 0
    running_loss = 0

    for e in range(epochs):
        # set the model in train mode to prevent the case if it has been set in eval mode
        start_time = time.time()
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            # set the grad to zero before the forward
            optimizer.zero_grad()
            # get the output of the model
            output = model.forward(images)
            # calculate the loss
            loss = criterion(output, labels)
            # do the back propagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # check the validation accuracy and loss if the validation_dataloader is set
                if validatation_dataloader is not None:
                    # set model in eval mode
                    model.eval()
                    # disable grad
                    with torch.no_grad():
                        accuracy, validation_loss = validate(model, validatation_dataloader, criterion, device)

                    # print status
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss / print_every),
                          "Validation test Loss: {:.4f}.. ".format(validation_loss / len(validatation_dataloader)),
                          "validation test Accuracy: {:.4f}".format(accuracy / len(validatation_dataloader)))

                    model.train()
                else:
                    # print status
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss / print_every))
                running_loss = 0
        print("Epoch run time {}".format(time.time() - start_time))


def validate(model, testloader, criterion, device=None):
    """Function to validate(test) a model

    Args:
        model (:nn.Module): The model use for validation
        testloader (:DataLoader): The training dataloader
        criterion : The loss function
        device (device): The device to use
    Returns:
        (accuracy, test_loss) : the accuracy and the test loss of the test
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        # output of the test batch
        output = model.forward(images)
        # calculate the test loss and increment the variable
        loss = criterion(output, labels)
        test_loss += loss.item()
        # probabilities since log soft_max is used we need to calculate the exp to get the probabilities
        ps = torch.exp(output)
        # get the highest prob
        predicted_classes = ps.max(dim=1)[1]
        compare_results = predicted_classes == labels.data
        accuracy += compare_results.type(torch.FloatTensor).mean()
    return accuracy, test_loss


def save_model(model, model_name, layers, train_dataset, optimizer, filepath):
    """ Save the model to a checkpoint file

    Args:
        model (:nn.Module): The model
        layers (dict of str): a dict of strings with  the information about the layers of the model
            layers = {
                "layer_name": last_layer_name,
                "input_size": input_size,
                "output_size": output_size,
                "hidden_layers": hidden_layers
            }
        model_name (str): The name of the model
        train_dataset (:Dataset):  The train dataset
        optimizer (:Optimizer): The optimizer
        filepath (str): The path of the saved file
    """

    checkpoint_data = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "model_layers": layers,
        "model_classifier_dropout": getattr(model, layers["layer_name"]).dropout.p,
        "model_class_to_idx": train_dataset.class_to_idx,
        "optimizer_state_dict": optimizer.state_dict,
        "optimizer_learning_rate": optimizer.param_groups[0]["lr"]
    }

    if os.path.exists(filepath):
        os.remove(filepath)
    torch.save(checkpoint_data, filepath)


def load_model(file_path, device):
    """ Rebuild the model from the checkpoint save file

    Args:
        file_path (string): The path of the saved file
        device (device): The device to load the data on
    Returns:
        (model, optimizer, layers): The rebuild model and optimizer and the layers dict from the parameters
    """
    checkpoint_data = torch.load(file_path, map_location=torch.device(device))
    layers = checkpoint_data["model_layers"]

    # create and load the state to the new model
    # get the pretrained network
    loaded_model = getattr(models, checkpoint_data["model_name"])(pretrained=True)

    # freeze all the parameters
    for param in loaded_model.parameters():
        param.requires_grad = False

    # create the classifier network and assign it to the last layer of the pretrained model
    classifier = ClassifierNetwork(layers["input_size"], layers["output_size"], layers["hidden_layers"],
                                   checkpoint_data["model_classifier_dropout"])
    setattr(loaded_model, layers["layer_name"], classifier)

    loaded_model.load_state_dict(checkpoint_data["model_state_dict"])
    loaded_model.class_to_idx = checkpoint_data["model_class_to_idx"]

    # create and load the state to the new optimizer
    loaded_optimizer = optim.Adam(getattr(loaded_model, layers["layer_name"]).parameters(),
                                  lr=checkpoint_data["optimizer_learning_rate"])
    loaded_optimizer.state_dict = checkpoint_data["optimizer_state_dict"]

    return loaded_model, loaded_optimizer, layers


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)

    # pytorch transforms can be used on and PIL.Image object
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])
    im = transform(im)
    return im


def predict(image_path, model_checkpoint, class_name_map, device=None, topk=1):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, optimizer, layers = load_model(model_checkpoint, device)

    model.to(device)
    image = process_image(image_path)
    image = image.to(device)
    # unsequeeze the image to match the input of the network
    image.unsqueeze_(0)

    # put model in eval mode
    model.eval()
    # disable grad
    with torch.no_grad():
        # output of the test batch
        output = model.forward(image)
        # probabilities since log soft_max is used we need to calculate the exp to get the probabilities
        ps = torch.exp(output)
    # copy the data to the cpu before converting to array
    ps = ps.cpu()
    # get the topk probabilities
    ps = ps.topk(topk)
    probs = ps.values.numpy()[0]
    indices = ps.indices.numpy()[0]

    # get the classes of the probs
    idx_to_class = {y: x for x, y in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]

    names = [class_name_map[x] for x in classes]

    return probs, classes, names


def get_random_images(images_dir, class_name_map, number=4):
    """ Get some random images from a folder
    Args:
        images_dir (string): The path of the images folder
        class_name_map (dict of str): The class to name mapping
        number (int): The number of the images to return
    Returns:
        images (list of tupes (image, label)): returns a list of tuples with the image path and the label
    """
    images_paths = []
    for path, dirs, files in os.walk(images_dir):
        for f in files:
            images_paths.append(os.path.normpath(path + '/' + f))

    images_map = np.array([(x, class_name_map[x.split(os.path.sep)[-2]]) for x in images_paths])

    idx = np.random.choice(len(images_map), number)

    return images_map[idx]
