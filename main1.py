import os  # dealing with directories
import matplotlib.pyplot as plt  # for visualizations
import numpy as np  # arrays
import pandas as pd  # for manipulating data
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from utils.helpers import create_transform, prepare_train_valid_test, unzip_input_file

# HYPERPARAMETERS
IMG_SIZE = 80
IMG_SIZE_ALEXNET = 227
SHOWN_IMAGE_COUNT = 64
columns = 8
rows = 8
hidden_size = 100
num_epochs = 1  # Updated to 1 epoch
batch_size = 32
learning_rate = 3e-5
UNZIP = False

BASE_DIR = os.getcwd()

# Our dataset class
class CustomDataset(Dataset):
    def __init__(self, arr, transform=None) -> None:
        self.x = [Image.fromarray(i[0], "RGB") for i in arr]
        self.y = np.array([i[1].argmax() for i in arr])
        self.transform = transform
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        y_label = self.y[index]
        if self.transform:
            img = self.transform(self.x[index])
        return img, y_label

    def __len__(self):
        return self.n_samples

# Declaring model
class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=4)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(self.relu(self.conv5(x)))
        x = self.dropout(x)
        x = x.reshape(-1, 256 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def training(train_loader, device, model, optimizer, epoch, criterion, training_loss_list, num_steps):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        training_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % num_steps == 0:
            print(f"train; epoch={epoch+1}, training loss = {np.round(loss.item(),4)}")

def evaluation(model, device, loader, criterion, validation_loss_list, epoch=None, validation=True):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        loss_all = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss_list.append(loss.item())
            loss_all += loss.item()
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        avg_loss = loss_all / len(loader)
        if validation:
            print(f"valid; epoch={epoch+1}, valid loss = {round(float(avg_loss),4)}, accuracy = {np.round(acc,4)} \n")
        else:
            print(f"test scores, valid loss = {round(float(avg_loss),4)}, accuracy = {np.round(acc,4)} \n")

def plot_loss_train_valid(training_loss_list, validation_loss_list):
    f, ax = plt.subplots(1, 2, figsize=(12, 3))
    pd.Series(training_loss_list).rolling(50).mean().plot(kind="line", title="Accuracy on CV data", ax=ax[0])
    pd.Series(validation_loss_list).rolling(50).mean().plot(kind="line", title="Loss on CV data", ax=ax[1])
    plt.subplots_adjust(wspace=0.8)
    ax[0].set_title("Loss on train data")
    ax[1].set_title("Loss on CV data")
    plt.show()

def get_test_preds(model, loader, device):
    model.eval()
    with torch.no_grad():
        test_classes = []
        test_preds = []
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            test_classes.append(labels)
            test_preds.append(predictions)
    test_classes = np.hstack([x.cpu().numpy() for x in test_classes])
    test_preds = np.hstack([x.cpu().numpy() for x in test_preds])
    return test_preds

def plot_some_preds(SHOWN_IMAGE_COUNT, columns, rows, test_preds, test_data):
    pred_labels = []
    for i in range(SHOWN_IMAGE_COUNT):
        r = test_preds[i]
        if r == 0:
            pred_labels.append("chair")
        elif r == 1:
            pred_labels.append("kitchen")
        elif r == 2:
            pred_labels.append("knife")
        elif r == 3:
            pred_labels.append("saucepan")

    shown_images = [x[0] for x in test_data[:SHOWN_IMAGE_COUNT]]
    fig = plt.figure(figsize=(20, 20))
    for m in range(1, columns * rows + 1):
        img = shown_images[m - 1].reshape([IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3])
        fig.add_subplot(rows, columns, m)
        plt.imshow(img)
        plt.title("Pred: " + pred_labels[m - 1])
        plt.axis("off")
    plt.show()

def main():
    if UNZIP:
        unzip_input_file("datasets.zip")

    train, cv, test_data = prepare_train_valid_test(
        BASE_DIR,
        "datasets",
        "train_data_mc.npy",
        "test_data_mc.npy",
        IMG_SIZE_ALEXNET,
        train_size=4800,
    )
    transform = create_transform()

    train_dataset = CustomDataset(train, transform)
    valid_dataset = CustomDataset(cv, transform)
    test_dataset = CustomDataset(test_data, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AlexNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader)
    training_loss_list = []
    validation_loss_list = []

    for epoch in range(num_epochs):
        training(
            train_loader,
            device,
            model,
            optimizer,
            epoch,
            criterion,
            training_loss_list,
            num_steps,
        )
        evaluation(
            model=model,
            device=device,
            loader=valid_loader,
            criterion=criterion,
            validation_loss_list=validation_loss_list,
            epoch=epoch,
            validation=True,
        )

    plot_loss_train_valid(training_loss_list, validation_loss_list)
    evaluation(
        model=model,
        device=device,
        loader=test_loader,
        criterion=criterion,
        validation_loss_list=validation_loss_list,
        validation=False,
    )
    test_preds = get_test_preds(model, test_loader, device)
    plot_some_preds(SHOWN_IMAGE_COUNT, columns, rows, test_preds, test_data)


if __name__ == "__main__":
    main()
