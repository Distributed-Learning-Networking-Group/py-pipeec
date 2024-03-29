from viztracer import VizTracer
import models.resnet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pypipeec import checkpoint, context
import torch.optim as optim
import torch.nn as nn


def run(model: torch.nn.Module, ckpter: checkpoint.CheckPointer, trainloader: DataLoader):

    ckpter.load_module()
    print(f"***** load checkpoint with timestamp *****")
    print(ckpter.timestamps())

    # Define the device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")

    # Move the model to the device
    model = model.to(device)

    ckpter.checkpoint_module_async()

    # Training loop
    for _ in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            ckpter.checkpoint_module_wait()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 100 == 99:    # print every 100 mini-batches
            # print("current loss {}' check pointing...".format(
            #     running_loss / 100))
            # running_loss = 0.0
            ckpter.checkpoint_module_async()

    print('Finished Training')


if __name__ == "__main__":
    model = models.resnet.resnet101(10, False, None)
    names = {i: f"worker_{i}" for i in range(8)}
    block_number = checkpoint.get_block_number(model)
    conf = context.NetworkConfig(8, 3, 3, {}, names, "mem", block_number)
    ckpter = checkpoint.CheckPointer("rs")

    # Define the transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the datasets
    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    # Create the dataloaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with VizTracer(output_file="output.json"):
        with ckpter.run_module_context(model, 0, conf, False):
            run(model, ckpter, trainloader)
