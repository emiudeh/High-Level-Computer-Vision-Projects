import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import numpy as np

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
layer_config= [512, 256]
num_classes = 10
num_epochs = 30
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg=0#0.001
num_training= 49000
num_validation =1000
fine_tune = True
pretrained= True

data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]#, transforms.RandomGrayscale(p=0.05)]
#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
# Q1,
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
        train=True,
        transform=norm_transform,
        download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
        train=False,
        transform=norm_transform
        )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



#pdb.set_trace()
class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        vgg_11_bn = models.vgg11_bn(pretrained)

        self.features  = vgg_11_bn.features;
        # Newly created modules have require_grad=True by default
        custom_classifier = []
        custom_classifier.append(nn.Linear(512,layer_config[0]))
        custom_classifier.append(nn.BatchNorm1d(layer_config[0]))
        custom_classifier.append(nn.ReLU())
        custom_classifier.append(nn.Linear(layer_config[0],layer_config[1]))
        custom_classifier.append(nn.BatchNorm1d(layer_config[1]))
        custom_classifier.append(nn.ReLU())
        custom_classifier.append(nn.Linear(layer_config[1],n_class))
        
        if fine_tune:
            set_parameter_requires_grad(self,fine_tune)
        self.classifier = nn.Sequential(*custom_classifier)
    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #pdb.set_trace()
        out=self.features(x)
        # out is 4d tensors, reshape it to 2d tensor
        out = out.view(batch_size,-1)
        out = self.classifier(out)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

# Initialize the model for this run
model= VggModel(num_classes, fine_tune, pretrained)

# Print the model we just instantiated
#pdb.set_trace()
print(model)

#################################################################################
# TODO: Only select the required parameters to pass to the optimizer. No need to#
# update parameters which should be held fixed (conv layers).                   #
#################################################################################

print("Params to learn:")
if fine_tune:
    params_to_update = []
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    params_to_update = model.classifier.parameters()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
else:
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

#pdb.set_trace()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)



# to track the training loss as the model trains
# losses of various model on each epoch
baseline_train_losses =[]
baseline_valid_losses =[]

# pre_trained_without_finetuned
pretrained_train_losses =[]
pretrained_valid_losses =[]
# pref_trained_with_finetuning
pretrained_finetuned_train_losses =[]
pretrained_finetuned_valid_losses =[]

current_epoch_train_loss = []
current_epoch_validation_loss =[]


n_epochs_stop = 5
min_val_loss = np.Inf
epochs_no_improve = 0

local_train_loss = [];
local_validation_loss = [];
import os
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        current_epoch_train_loss.append(loss.item())
        
           

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs,labels)
            current_epoch_validation_loss.append(loss.item())
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #################################################################################
        # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
        # the model which has acheieved the best validation accuracy so-far.            #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # If the validation loss is at a minimum
        val_loss = 100*(correct/float(total))
        if val_loss < min_val_loss:
          # Save the model
          torch.save(model.state_dict(), 'best_model.ckpt')
          epochs_no_improve = 0
          min_val_loss = val_loss
        else:
          epochs_no_improve += 1
          # Check early stopping condition
          if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print('Validataion accuracy is: {} %'.format(100 * correct / total))
    # compute loss every epoch 
    #import pdb
    #pdb.set_trace()
    avg_train_loss = np.average(current_epoch_train_loss)
    avg_validation_loss = np.average(current_epoch_validation_loss)
    
    local_train_loss.append(avg_train_loss);
    local_validation_loss.append(avg_validation_loss)
    if(pretrained and fine_tune  ):
      pretrained_finetuned_train_losses.append(avg_train_loss)
      pretrained_finetuned_valid_losses.append(avg_validation_loss)
    if(pretrained and not fine_tune):
      pretrained_train_losses.append(avg_train_loss)
      pretrained_valid_losses.append(avg_validation_loss)
    if( (not pretrained) and (not fine_tune)):
      baseline_train_losses.append(avg_train_loss)
      baseline_valid_losses.append(avg_validation_loss)
    current_epoch_train_loss = []
    current_epoch_validation_loss = []

    
#Dump the training va{lues
import pickle
file_name=""
if(pretrained and fine_tune):
  file_name="pretrained_finetune.txt"    
if(pretrained and not fine_tune):
  file_name="pretrained.txt"
if( (not pretrained) and (not fine_tune)):
  file_name="baseline.txt"
with open(file_name, "wb") as fp:   #Pickling
      pickle.dump(local_train_loss,fp)
      pickle.dump(local_validation_loss,fp)
      
#################################################################################
# TODO: Use the early stopping mechanism from previous question to load the     #
# weights from the best model so far and perform testing with this model.       #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
best_model = model.load_state_dict(torch.load('best_model.ckpt'))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

