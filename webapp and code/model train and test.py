#First we must import the libraries which are needed.
import time
import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models,transforms
import numpy as np
import copy
import matplotlib.pyplot as pyplot
from sklearn.metrics import f1_score
#completed our import process.

#enable the gpu(only nvdia supported) if it exists
device=(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(device)

#now , we define the transformations to be applied to our data befor loading it.
print("test1")
transformer={
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),#changes pixel range from 0-255 to 0-1(tensor form,accepted by pytorch.)
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
	'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

#now we load data and apply transformations.
datafolder=['train','val','test']
path='Dataset'
image_datasets={img: datasets.ImageFolder(os.path.join(path,img),transform=transformer[img])
for img in datafolder} #this is a shorcut like list comprehensions to initialize a dictionary. Image_datasets is a dictionary with key as foldername like train, val, test and value as the list of transformed images in these folders.

#dataloader is used to feed the data to model in batches to avoid overloading during training.
dl={img:torch.utils.data.DataLoader(image_datasets[img],batch_size=4,shuffle=True,num_workers=0)
for img in datafolder} # here d1 is a dictionary with folder name as key and image batches as value

#fetch number of images.
size={img: len(image_datasets[img]) for img in datafolder}
print(size)
#alternate way to fetch the number of images
#size={}
#for img in datafolder:
    #size[img]=len(image_datasets[img])
#print(size)

#fetch class names(labels) from train folder only
class_names=image_datasets['train'].classes
print(class_names)

#here using transfer learning. i.e. build a model based on a pretrained model.
model_resnet=models.resnet18(pretrained=True)
model_vgg16=models.vgg16(pretrained=True)
model_alexnet=models.alexnet(pretrained=True)
model_googlenet= models.googlenet(pretrained=True)

# now we enable the gpu for the model
model_resnet=model_resnet.to(device)
model_vgg16=model_vgg16.to(device)
model_alexnet=model_alexnet.to(device)
model_googlenet=model_googlenet.to(device)

#set the loss function
c=nn.CrossEntropyLoss()

#now we create a function to train models.
def train_M(m,c,o,s,n=25):#m=model,c=criteria,o=optimizer,s=schedular,n=no.of epochs.
    since=time.time()
    best_M=copy.deepcopy(m.state_dict())
    best_A=0.0
    for epoch in range(n):
        print(f"epoch number: {epoch}/{n-1}")
        print("####"*10)
        for phase in ['train', 'val']:
            if phase=='train':
                m.train()
            else:
                m.eval()
            run_loss=0.0
            run_correct=0
            model_resnet.to(device)
            m.to(device)


            for inputs,labels in dl[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                o.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs=m(inputs)
                    _, preds=torch.max(outputs,1)
                    loss=c(outputs,labels)

                    if phase=='train':
                        loss.backward()
                        o.step()

                run_loss+=loss.item()*inputs.size(0)
                run_correct+=torch.sum(preds==labels.data)

            if phase=='train':
                s.step()

            epoch_loss=run_loss/size[phase]
            epoch_acc=run_correct.double()/size[phase]

            print(f"{phase} loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

            if phase=='val' and epoch_acc>best_A:
                best_A=epoch_acc
                best_M=copy.deepcopy(m.state_dict())

        print()

    t=time.time()-since
    print(f"Model training done in :{t//60:.0f} minutes and {t%60:.0f} seconds")
    print(f"Best Accuracy: {best_A:.4f}")

    m.load_state_dict(best_M)
    return m

#Ensure that gradient is not calculated for every model.


for para in model_resnet.parameters():
    para.requires_grad=False

for para in model_vgg16.parameters():
    para.requires_grad=False

for para in model_alexnet.parameters():
    para.requires_grad=False

for para in model_googlenet.parameters():
    para.requires_grad=False


# now we change the final layer of each model as per our data set...
#final layer is different for each model so proceed carefully.....

num=model_resnet.fc.in_features
model_resnet.fc=nn.Linear(num,len(class_names))

num=model_vgg16.classifier[6].in_features
model_vgg16.classifier[6]=nn.Linear(num,len(class_names))

num=model_alexnet.classifier[6].in_features
model_alexnet.classifier[6]=nn.Linear(num,len(class_names))

num=model_googlenet.fc.in_features
model_googlenet.fc=nn.Linear(num,len(class_names))

nb_classes = 4
#time to train the models
"""opt=optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)
s=lr_scheduler.StepLR(opt,step_size=7,gamma=0.1)
model_resnet=train_M(model_resnet,c,opt,s,n=25)
target = './t_resnet18.pth'
torch.save(model_resnet.state_dict(), target)

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dl['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_resnet(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)"""

"""opt=optim.SGD(model_vgg16.classifier[6].parameters(), lr=0.001, momentum=0.9)
s=lr_scheduler.StepLR(opt,step_size=7,gamma=0.1)
model_vgg16=train_M(model_vgg16,c,opt,s)
target='./t_vgg16.pth'
torch.save(model_vgg16.state_dict(),target)

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dl['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_vgg16(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)"""


"""opt=optim.SGD(model_alexnet.classifier[6].parameters(), lr=0.001, momentum=0.9)
s=lr_scheduler.StepLR(opt,step_size=7,gamma=0.1)
model_alexnet=train_M(model_alexnet,c,opt,s)
target='./t_alexnet.pth'
torch.save(model_alexnet.state_dict(),target);

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dl['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_alexnet(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)"""

opt=optim.SGD(model_googlenet.fc.parameters(), lr=0.001, momentum=0.9)
s=lr_scheduler.StepLR(opt,step_size=7,gamma=0.1)
model_googlenet=train_M(model_googlenet,c,opt,s,n=25)
target = './t_googlenet.pth'
torch.save(model_googlenet.state_dict(), target)
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dl['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_googlenet(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

