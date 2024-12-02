#Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import torchvision.models as models
import torchvision.transforms as transforms
from data_loader import DataLoaderCreator

import smdebug.pytorch as smd
# from smdebug import modes

# from tqdm import tqdm
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

def test(model, test_loader, criterion, device):
    '''
    This function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

    total_loss /= len(test_loader.dataset)
    print(
            "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                total_loss, correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset)
            )
        )

def train(model, train_loader, criterion, optimizer, epoch, device):
    '''
    This function that can take a model and
    data loaders for training and will get train the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    epoch,
                    i * len(inputs),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
    return model
    
def net(num_classes, pretrained=True):
    '''
    This function that initializes your model
    Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(num_classes)
    )

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):

    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = args.num_classes
    
    '''
    Initialize a model by calling the net function
    '''
    model=net(args.num_classes).to(device)

    # Initialize DataLoaderCreator
    train_dataset_directory = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    valid_dataset_directory = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    test_dataset_directory = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
    data_loader_creator = DataLoaderCreator(
        img_size=(224, 224),
        batch_size=args.batch_size,
        train_dir=train_dataset_directory,
        val_dir = valid_dataset_directory,
        test_dir = test_dataset_directory
    )
    data_loaders = data_loader_creator.create_data_loaders()
    
    '''
    Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    print("Starting training...")
    for epoch in range(1, args.num_epochs+1):
        model = train(model, data_loaders["train"], loss_criterion, optimizer, epoch, device)
    
        '''
        Test the model to see its accuracy
        '''
        test(model, data_loaders["val"], loss_criterion, device)

    # Test the model
    print("Testing the model...")
    test(model, data_loaders["test"], loss_criterion, device)
    
    '''
    Save the trained model
    '''
    # Save the trained model
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "model.pt")
    model = model.to(device)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(model_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders")
    parser.add_argument("--num_classes", type=int, default=133, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    
    args=parser.parse_args()
    
    main(args)
