# -*- coding: utf-8 -*-
import copy
import os
import torch
from torch import optim
from dataset import get_dataloader
from models import ResNet18Model, ConvNextTinyModel
from training import train


def main(dataset_path, model_name, epochs, weights=None, with_ema = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_loader, validation_loader, class_label_mapping = get_dataloader(dataset_path)

    num_classes = len(class_label_mapping)

    if model_name == "ResNet18Model":
        model = ResNet18Model(num_classes=num_classes)
    elif model_name == "ConvNextTinyModel":
        model = ConvNextTinyModel(num_classes=num_classes)
    else:
        raise ValueError("Does not exist!")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, optimizer, training_loader, validation_loader, device, class_label_mapping, epochs, load_model_path = weights)

if __name__ == '__main__':
    print("Current Directory: ", os.getcwd())
    
    img_folder = os.path.join("..", "datasets", "simpsons", "imgs")
    epochs = 30

    #Training of ResNet18Model
    main(dataset_path=img_folder, model_name="ResNet18Model", epochs=epochs)

    #Training of ConvNextModel
    main(dataset_path=img_folder, model_name="ConvNextTinyModel", epochs=epochs)