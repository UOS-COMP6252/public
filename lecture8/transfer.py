
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
import copy


seed=9797 
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True


# ### Create custom dataset class
# 
# - Since we use ```random_split``` we cannot use different transforms to the training and validation datasets
# - Dataset is an abstract class. It cannot be instantiated 
# - We need to wrap the datasets obtained from ```random_split``` with a custom dataset class

class MyDataset(Dataset):
    def __init__(self,subset,transform=None):
        self.subset=subset
        self.transform=transform
    def __getitem__(self,idx):
        x,y=self.subset[idx]
        if self.transform:
            x=self.transform(x)
        return x,y
    def __len__(self):
        return len(self.subset)


# #### Data augmentation
# 

augment_choices={'trivial':transforms.TrivialAugmentWide(),'random':transforms.RandAugment(),
                 'mix':transforms.AugMix(),'auto':transforms.AutoAugment()}
aug='trivial'


data_transforms = {
     'train':  transforms.Compose([ transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1., 1., 1.])
    ]) if aug=='None' else
     
     transforms.Compose([ transforms.TrivialAugmentWide(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1., 1., 1.])
    ]), 
  
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1., 1., 1.])
    ])
}



dataset=torchvision.datasets.ImageFolder("flowers")
class_names=dataset.classes
print(class_names)
fig=plt.figure()
fig.tight_layout()
plt.subplots_adjust( wspace=.1, hspace=.3)
for i in range(20):
            #img,label=next(itr)
            img,label=dataset[i*200]
            t=fig.add_subplot(4,5,i+1)
            # set the title of the image equal to its label
            t.set_title(class_names[label])
            t.axes.get_xaxis().set_visible(False)
            t.axes.get_yaxis().set_visible(False)
            plt.imshow(img)



train_d,valid_d=random_split(dataset,lengths=[0.8,0.2])

datasets={'train':train_d,'val':valid_d}
image_datasets = {x: MyDataset(datasets[x],data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True if x=='train' else False, num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def evaluate(model, criterion):
    with torch.no_grad():
        model.eval()   
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.double() / dataset_sizes['val']
    return epoch_loss,epoch_acc


def train_model(model, criterion, optimizer,scheduler=None, num_epochs=100):

    for epoch in range(num_epochs):
        model.train()  
        running_loss,running_corrects =0.0, 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
       
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if scheduler != None:
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        v_loss,v_acc=evaluate(model,criterion)
        writer.add_scalars("loss",{'train':epoch_loss,'valid':v_loss},epoch)
        writer.add_scalars("acc",{'train':epoch_acc,'valid':v_acc},epoch)

        if  v_acc > best_acc:
                best_acc = v_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch%5==0:
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            print("t_loss={:.4f},t_acc={:.4f}".format(epoch_loss,epoch_acc))
            print("v_loss={:.4f},v_acc={:.4f}".format(v_loss,v_acc))
    ## training is done. Return the model with the best validation accuracy
    model.load_state_dict(best_model_wts)
    return model


import timm


#model_name='resnet18'
model_name='deit_base_patch16_224'
model=timm.create_model(model_name=model_name,pretrained=True)

for p in model.parameters():
    p.requires_grad=False

model.reset_classifier(len(class_names))
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ## Using hub


#model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
for p in model.parameters():
    p.requires_grad=False

#nfeatures = model.fc.in_features
#model.fc = nn.Linear(nfeatures, len(class_names))
model.reset_classifier(len(class_names))
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


model= train_model(model, criterion, optimizer,scheduler=scheduler,num_epochs=50)

