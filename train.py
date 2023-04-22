import torch
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from net import Net, transform, test_transform, net_batch_size, net_lr, net_criterion, net_optimizer

trainset = datasets.ImageFolder('./data/traindata', transform=transform)
validationset = datasets.ImageFolder('./data/validationdata', transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=net_batch_size, shuffle=True)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=len(validationset), shuffle=False)

# create neural network
net = Net()
criterion = net_criterion()
optimizer = net_optimizer(net.parameters(), lr=net_lr)
batches = len(trainloader)

# loop over the dataset
loss_list = []
training_acc_list = []
validation_acc_list = []
epoch = 0

# constantly train the data
while True:
    try:
        loss_total = 0
        acc_total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, 1)[1]
            acc = torch.eq(predicted, labels)
            loss_total += loss
            acc_total += acc.sum() / len(acc)
        
        
        # get accuracy with validation set
        with torch.no_grad():
            data, labels = next(iter(validationloader))
            outputs = net(data)
            predicted = torch.max(outputs, 1)[1]
            correct = torch.eq(predicted, labels).sum()
            
            validation_acc = correct / len(validationset)

        epoch_loss = loss_total/batches
        epoch_training_acc = acc_total/batches
        epoch_validation_acc = validation_acc

        loss_list.append(epoch_loss.detach())
        training_acc_list.append(epoch_training_acc)
        validation_acc_list.append(epoch_validation_acc)

        print("Epoch {0} Loss {1} Training Acc {2} Validation Acc {3}".format(
            epoch + 1, epoch_loss, epoch_training_acc, epoch_validation_acc
        ))
        
        torch.save(net.state_dict(), f'./models/model{epoch}.pth')
        epoch += 1
    
    except KeyboardInterrupt:
        # plot loss data
        plt.subplot(1, 2, 1)
        plt.title("Training Loss"); plt.xlabel("epochs"); plt.ylabel("loss")
        plt.plot(loss_list,'r')

        # plot accuracy data
        plt.subplot(1, 2, 2)
        plt.title("Accuracy"); plt.xlabel("epochs"); plt.ylabel("acc")
        plt.plot(training_acc_list,'r', label="Training Set Accuracy")
        plt.plot(validation_acc_list,'b', label="Validation Set Accuracy")
        plt.legend()
        plt.show()