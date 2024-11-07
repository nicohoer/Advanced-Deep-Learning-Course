# -*- coding: utf-8 -*-
import copy
import os
from datetime import datetime
from sys import prefix
from typing import List
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train(model, optimizer, train_loader, val_loader, device, class_names, epochs, load_model_path = None):
    loss_criterion = torch.nn.CrossEntropyLoss()
    training_start_epoch = 0
    best_validation_accuracy = 0

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    if load_model_path:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    for epoch in range(training_start_epoch, epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = loss_criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        class_accuracies = evaluation(model, val_loader, class_names, device)

        total_accuracy = np.mean(class_accuracies)
        print("Accuracy of Model: ", total_accuracy)


        if total_accuracy > best_validation_accuracy:
            best_validation_accuracy = total_accuracy
            if load_model_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, os.path.join(load_model_path, 'Best', f'best_{model.__class__.__name__}.pt'))


def evaluation(model, val_loader, classes, device):
    model.eval()
    correct_classes = list(0. for i in range(len(classes)))
    total_classes = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            is_correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                correct_classes[label] += is_correct[i].item()
                total_classes[label] += 1

    class_accuracies = [correct * 100.0 / total if total != 0.0 else np.nan for correct, total in zip(correct_classes, total_classes)]
    return class_accuracies