# -*- coding: utf-8 -*-

if __name__ == '__main__':
    from tqdm.notebook import tqdm
    import pickle
    import torch.optim as optim
    import torch.nn.functional as F
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision
    import torch
    import wandb
    import argparse
    wandb.login()

    wandb.init(
        # set the wandb project where this run will be logged
        project="MVC_Lab_week1",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.01,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 20,
        }
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # transform before training
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load train, valid, test data
    trainset, validset = torch.utils.data.random_split(
        torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform),
        lengths=[.8, .2])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=16, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=4)

    dict1 = {}
    with open("./data/cifar-100-python/meta", 'rb') as fo:
        dict1 = pickle.load(fo, encoding='latin1')

    classes = dict1['fine_label_names']

    # Your deep neural network

    class RestNetBasicBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            super(RestNetBasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            output = self.conv1(x)
            output = F.relu(self.bn1(output))
            output = self.conv2(output)
            output = self.bn2(output)
            return F.relu(x + output)

    class RestNetDownBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            super(RestNetDownBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        def forward(self, x):
            conv1x1 = self.conv1x1(x)
            output = self.conv1(x)
            out = F.relu(self.bn1(output))

            out = self.conv2(out)
            out = self.bn2(out)
            return F.relu(conv1x1 + out)

    class RestNet18(nn.Module):
        def __init__(self):
            super(RestNet18, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = nn.Sequential(RestNetBasicBlock(
                64, 64), RestNetBasicBlock(64, 64))
            self.layer2 = nn.Sequential(RestNetDownBlock(
                64, 128), RestNetBasicBlock(128, 128))
            self.layer3 = nn.Sequential(RestNetDownBlock(
                128, 256), RestNetBasicBlock(256, 256))
            self.layer4 = nn.Sequential(RestNetDownBlock(
                256, 512), RestNetBasicBlock(512, 512))
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.fc = nn.Linear(512, 100)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.reshape(x.shape[0], -1)
            out = self.fc(out)
            return out

    resnet = RestNet18().to(device)

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

    # start training

    for epoch in range(20):  # epoch
        print('now epoch: ', epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader):

            inputs, labels = data[0].to(device), data[1].to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            running_loss += loss.item()

            if i % 250 == 0:
                # validation
                resnet.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for data in validloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = resnet(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('accuracy:', 100 * correct / total)
                wandb.log({"accuracy": correct/total})
                resnet.train()

            running_loss = 0.0

    print('Finished Training')

    # test network on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # calculate accuracy
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(100):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    wandb.finish()
