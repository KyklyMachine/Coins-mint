import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

data_dir = r'C:\Users\Nikolay\Desktop\PrPr\CNN\data'

train_dir = data_dir + '//train'
valid_dir = data_dir + '//validation'
test_dir = data_dir + '//test'

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.RandomRotation(45),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            norm_mean,
                                            norm_std
                                        )
                                      ])

valid_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(
                                            norm_mean,
                                            norm_std
                                        )
                                      ])

test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            norm_mean,
                                            norm_std
                                        )
                                     ])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=60, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=60, shuffle=True)

dataloaders = {'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader}

def imshow_numpy(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
        
    ax.grid(False)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)

import json

with open(r'C:\Users\Nikolay\Desktop\PrPr\CNN\cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_resnet = models.resnet34(pretrained=False)

# Freeze parameters in pre trained ResNET
#for param in model_resnet152.parameters():
    #param.requires_grad = False

out_classes = len(cat_to_name)

model_resnet.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(p=0.7),
    nn.Linear(512, out_classes)
)

# Check the modified fc layer
print(model_resnet.fc)

# Pre-trained  resnet34
model_resnet34 = models.resnet34(pretrained=True)

# Freeze parameters in pre trained ResNET
#for param in model_resnet34.parameters():
    #param.requires_grad = False

out_classes = len(cat_to_name)

model_resnet34.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(p=0.7),
    nn.Linear(512, out_classes)
)

# Check the modified fc layer
print(model_resnet34.fc)

is_GPU_available = torch.cuda.is_available()

if is_GPU_available:
    device = 'cuda'
    print('training on GPU')
else:
    device = 'cpu'
    print('training on CPU')

# Choose the model I want to choose
my_model = model_resnet34

my_model.to(device)

my_model.class_to_idx = train_dataset.class_to_idx

def save_model(model, val_loss):
    model = {
        'state_dict': model.state_dict(),
        'fc': model.fc,
        'min_val_loss': val_loss,
        'class_to_idx': model.class_to_idx,
    }
    
    torch.save(model, 'checkpoint_cnn_resnet34.pth')

def load_checkpoint_resnet152(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet152(pretrained=True)
    
    # Freeze parameters (in case we want to train more)
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

my_model = my_model
epochs = 100
lr = 0.001
criterion = nn.CrossEntropyLoss()
min_loss = np.Inf

def train (my_model, criterion, epochs = 15, lr=0.001, min_valid_loss=np.Inf):
    best_model = my_model
    optimizer = optim.SGD(my_model.parameters(), lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True,
                                                     patience=5, min_lr=0.00001)
    
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0
        
        my_model.train()
        for images, labels in dataloaders['train_loader']:
            optimizer.zero_grad()

            # Move tensors to GPU if available
            inputs, labels = images.to(device), labels.to(device)

            # Forward pass
            output = my_model(inputs)

            loss = criterion(output, labels)
            loss.backward()

            # Update weights
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        my_model.eval()
        for inputs, labels in dataloaders['valid_loader']:
            # Move tensors to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            output = my_model(inputs)

            val_loss = criterion(output, labels)
            
            valid_loss += val_loss.item() * inputs.size(0)
            
            # Accuracy
            _, predicted = output.topk(1, dim=1)
            equals = predicted == labels.view(*predicted.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        # Calculate the losses
        train_loss = train_loss/len(dataloaders["train_loader"].dataset)
        valid_loss = valid_loss/len(dataloaders["valid_loader"].dataset)
        accuracy = (accuracy/len(dataloaders["valid_loader"]))*100
        
        # Update lr
        scheduler.step(valid_loss)
        
        print('Epoch {}'.format(epoch + 1))
        print('Train loss: {0:.2f}'.format(train_loss))
        print('Valid loss: {0:.2f}'.format(valid_loss))
        print('Accuracy: {0:.2f}%'.format(accuracy))
        
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            save_model(my_model, valid_loss)
            # best_model stores the model with the lowest valid loss
            best_model = my_model
            print('Valid loss has decreased. Saving model...')
        
        print('--------------------------------------------')
    return best_model

my_model = train(my_model=my_model, criterion=criterion, epochs=epochs, lr=lr, min_valid_loss=min_loss)
