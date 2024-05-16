import cv2
import os
from tqdm import tqdm
import pandas as pd

from glob import glob
from sklearn.model_selection import train_test_split
import shutil

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim import lr_scheduler
from tqdm import tqdm

import torch
import numpy as np
import random
import os

from glob import glob
from sklearn.model_selection import train_test_split
import shutil

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def del_empty_images():
    MIN_CONTOUR_AREA = 1000  # adjust this value as needed
    df = pd.DataFrame()

    for i in ['92', '95', '98']:
        directory = f'output_data/{i}' # папка, созданная в задании 2
        if not os.path.exists(f'images/{i}'): # выходная папка
            os.makedirs(f'images/{i}')
        for file in tqdm(os.listdir(directory)):
            frame = cv2.imread(directory + '/' + file)
            frame_h, frame_w = frame.shape[:2]
            frame = cv2.flip(frame, 0)
            blured_frame = cv2.blur(frame, (7, 7), 0)
            binary = cv2.inRange(blured_frame, (0, 0, 0), (100, 80, 80))
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                    cv2.imwrite(f'images/{i}/{file}', frame)

def create_dataset_structure(base_path, categories):
    for category in categories:
        os.makedirs(os.path.join(base_path, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'val', category), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'test', category), exist_ok=True)

def distribute_files(src_folder, dest_folder, categories):
    for category in categories:
        files = glob(os.path.join(src_folder, category, '*'))
        train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
        
        for file in train_files:
            shutil.copy(file, os.path.join(dest_folder, 'train', category))
        for file in val_files:
            shutil.copy(file, os.path.join(dest_folder, 'val', category))
        for file in test_files:
            shutil.copy(file, os.path.join(dest_folder, 'test', category))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # для мульти-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, criterion, optimizer, num_epochs=3):
    # If CUDA is available, use the GPU, else revert to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to the specified device
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders['train']):
            # Move inputs and labels to the same device as the model
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Acc: {epoch_acc}')

    return model

def calculate_metrics(model, dataloader, device, model_name):
    model.eval()  # Переключение модели в режим оценки
    true_labels = []
    predictions = []

    with torch.no_grad():  # Отключение градиентов
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().view(-1).tolist())
            true_labels.extend(labels.cpu().view(-1).tolist())

    # Вычисление точности, полноты и F1-меры
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print(f'Precision: {round(precision, 2)}')
    print(f'Recall: {round(recall, 2)}')
    print(f'F1 Score: {round(f1, 2)}')

    return pd.concat([df, pd.DataFrame({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }, index=[0])], ignore_index=True)

source_folder = 'images'
dataset_folder = 'dataset'
categories = ['92', '95', '98']

del_empty_images() # удаляем картинки без пузырьков
create_dataset_structure(dataset_folder, categories) # создаем структуру датасета 
distribute_files(source_folder, dataset_folder, categories) # записываем файлы

# Фиксируем сиды

seed = 42
set_seed(seed)

data_dir = 'dataset'
df = pd.DataFrame()

# Transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}

num_classes = len(image_datasets['train'].classes) # количество классов

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Models

vgg11 = models.vgg11(pretrained=True)
resnet152 = models.resnet152(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
vgg13 = models.vgg13(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
vgg19 = models.vgg19(pretrained=True)


# Params (off gradients)

for param in vgg11.parameters():
    param.requires_grad = False

for param in resnet152.parameters():
    param.requires_grad = False

for param in resnet101.parameters():
    param.requires_grad = False

for param in resnet18.parameters():
    param.requires_grad = False

for param in resnet34.parameters():
    param.requires_grad = False

for param in vgg13.parameters():
    param.requires_grad = False

for param in vgg16.parameters():
    param.requires_grad = False

for param in vgg19.parameters():
    param.requires_grad = False

# Change output layer

vgg11.classifier[6] = nn.Linear(vgg11.classifier[6].in_features, num_classes)
resnet152.fc = nn.Linear(resnet152.fc.in_features, num_classes)
resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
vgg13.classifier[6] = nn.Linear(vgg13.classifier[6].in_features, num_classes)
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)
vgg19.classifier[6] = nn.Linear(vgg19.classifier[6].in_features, num_classes)


# Criterions and optimizers

criterion_vgg11 = nn.CrossEntropyLoss()
optimizer_vgg11 = optim.Adam(vgg11.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_vgg11 = lr_scheduler.StepLR(optimizer_vgg11, step_size=100, gamma=0.1)

criterion_resnet152 = nn.CrossEntropyLoss()
optimizer_resnet152 = optim.Adam(resnet152.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_resnet152 = lr_scheduler.StepLR(optimizer_resnet152, step_size=100, gamma=0.1)

criterion_resnet101 = nn.CrossEntropyLoss()
optimizer_resnet101 = optim.Adam(resnet101.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_resnet101 = lr_scheduler.StepLR(optimizer_resnet101, step_size=100, gamma=0.1)

criterion_resnet34 = nn.CrossEntropyLoss()
optimizer_resnet34 = optim.Adam(resnet34.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_resnet34 = lr_scheduler.StepLR(optimizer_resnet34, step_size=100, gamma=0.1)

criterion_resnet18 = nn.CrossEntropyLoss()
optimizer_resnet18 = optim.Adam(resnet18.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_resnet18 = lr_scheduler.StepLR(optimizer_resnet18, step_size=100, gamma=0.1)

criterion_vgg13 = nn.CrossEntropyLoss()
optimizer_vgg13 = optim.Adam(vgg13.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_vgg13 = lr_scheduler.StepLR(optimizer_vgg13, step_size=100, gamma=0.1)

criterion_vgg16 = nn.CrossEntropyLoss()
optimizer_vgg16 = optim.Adam(vgg16.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_vgg16 = lr_scheduler.StepLR(optimizer_vgg16, step_size=100, gamma=0.1)

criterion_vgg19 = nn.CrossEntropyLoss()
optimizer_vgg19 = optim.Adam(vgg19.parameters(), lr=0.001, betas=(0.9, 0.999))
exp_lr_scheduler_vgg19 = lr_scheduler.StepLR(optimizer_vgg19, step_size=100, gamma=0.1)

# To device

vgg11 = vgg11.to(device)
resnet152 = resnet152.to(device)
resnet101 = resnet101.to(device)
resnet18 = resnet18.to(device)
resnet34 = resnet34.to(device)
vgg13 = vgg13.to(device)
vgg16 = vgg16.to(device)
vgg19 = vgg19.to(device)

# Train

vgg11 = train_model(vgg11, criterion_vgg11, optimizer_vgg11, num_epochs=1)
resnet152 = train_model(resnet152, criterion_resnet152, optimizer_resnet152, num_epochs=5)
resnet101 = train_model(resnet101, criterion_resnet101, optimizer_resnet101, num_epochs=1)
resnet18 = train_model(resnet18, criterion_resnet18, optimizer_resnet18, num_epochs=1)
resnet34 = train_model(resnet34, criterion_resnet34, optimizer_resnet34, num_epochs=1)
vgg13 = train_model(vgg13, criterion_vgg13, optimizer_vgg13, num_epochs=1)
vgg16 = train_model(vgg16, criterion_vgg16, optimizer_vgg16, num_epochs=1)
vgg19 = train_model(vgg19, criterion_vgg19, optimizer_vgg19, num_epochs=1)

# Save models

torch.save(vgg11, 'vgg11.pth')
torch.save(resnet152, 'resnet152.pth')
torch.save(resnet101, 'resnet101.pth')
torch.save(resnet18, 'resnet18.pth')
torch.save(resnet34, 'resnet34.pth')
torch.save(vgg13, 'vgg13.pth')
torch.save(vgg16, 'vgg16.pth')
torch.save(vgg19, 'vgg19.pth')

# Calculating metrics

df = calculate_metrics(vgg11, dataloaders['val'], device, model_name='VGG11')
df = calculate_metrics(resnet152, dataloaders['val'], device, model_name='resnet152')
df = calculate_metrics(resnet101, dataloaders['val'], device, model_name='resnet101')
df = calculate_metrics(resnet18, dataloaders['val'], device, model_name='resnet18')
df = calculate_metrics(resnet34, dataloaders['val'], device, model_name='resnet34')
df = calculate_metrics(vgg13, dataloaders['val'], device, model_name='vgg13')
df = calculate_metrics(vgg16, dataloaders['val'], device, model_name='vgg16')
df = calculate_metrics(vgg19, dataloaders['val'], device, model_name='vgg19')

# val - не валидационная выборка, а тестовая, забыл переименовать

df.to_csv('output_metrics_nn.csv', index=False)
