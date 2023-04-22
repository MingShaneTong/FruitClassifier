import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from net import Net, test_transform

testset = datasets.ImageFolder('./data/testdata', transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=len(testset))

net = Net()
net.load_state_dict(torch.load("./model.pth"))
net.eval()

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    data, labels = next(iter(testloader))
    outputs = net(data)
    predicted = torch.max(outputs, 1)[1]
    correct = torch.eq(predicted, labels).sum()
    print(f'Accuracy: {100 * correct / len(testset)} %')