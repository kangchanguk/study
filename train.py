import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
#from fastai.vision import Path
import torch
from torch.autograd import Variable

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALL_CHAR_SET = NUMBER
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

def encode(a):
    onehot = [0]*ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] += 1
    return onehot


class Mydataset(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.path = path
        if is_train:
            self.img = os.listdir(self.path)[:1000]
        else:
            self.img = os.listdir(self.path)[1001:]


        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img[idx]

        img = Image.open(self.path + "/" + img_path)
        img = img.convert('L')
        label =  str(img_path)[:-5]
        label_oh = []
        for i in label:
            label_oh += encode(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label

    def __len__(self):
        return len(self.img)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    transform_train_1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop((22, 22)),
        transforms.ToTensor(),
    ])


    transform_train_2 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_train_3 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])

    transform_train_4 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])

    train_ds = Mydataset('./captcha', transform=transform)
    train_ds_1 = Mydataset('./captcha', transform=transform_train_1)
    train_ds_2 = Mydataset('./captcha', transform=transform_train_2)
    train_ds_3 = Mydataset('./captcha', transform=transform_train_3)
    train_ds_4 = Mydataset('./captcha', transform=transform_train_4)
    test_ds = Mydataset('./captcha', False, transform)
    train_dl = DataLoader(train_ds, batch_size=64, num_workers=0)
    train_dl_1 = DataLoader(train_ds_1, batch_size=64, num_workers=0)
    train_dl_2 = DataLoader(train_ds_2, batch_size=64, num_workers=0)
    train_dl_3 = DataLoader(train_ds_3, batch_size=64, num_workers=0)
    train_dl_4 = DataLoader(train_ds_4, batch_size=64, num_workers=0)
    test_dl = DataLoader(train_ds, batch_size=1, num_workers=0)
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN * MAX_CAPTCHA, bias=True)

    loss_func = nn.MultiLabelSoftMarginLoss()
    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        for step, i in enumerate(train_dl):
            img, label_oh, label = i
            img = Variable(img)
            label_oh = Variable(label_oh.float())
            pred = model(img)
            loss = loss_func(pred, label_oh)
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

        for step, i in enumerate(train_dl_1):
            img, label_oh, label = i
            img = Variable(img)
            label_oh = Variable(label_oh.float())
            pred = model(img)
            loss = loss_func(pred, label_oh)
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

        for step, i in enumerate(train_dl_2):
            img, label_oh, label = i
            img = Variable(img)
            label_oh = Variable(label_oh.float())
            pred = model(img)
            loss = loss_func(pred, label_oh)
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

        for step, i in enumerate(train_dl_3):
            img, label_oh, label = i
            img = Variable(img)
            label_oh = Variable(label_oh.float())
            pred = model(img)
            loss = loss_func(pred, label_oh)
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

        for step, i in enumerate(train_dl_4):
            img, label_oh, label = i
            img = Variable(img)
            label_oh = Variable(label_oh.float())
            pred = model(img)
            loss = loss_func(pred, label_oh)
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

        if epoch % 5 == 0:
            correct = 0
            total_size = 0
            for step, (img, label_oh, label) in enumerate(test_dl):
                img = Variable(img)
                pred = model(img)

                c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
                c1 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN * 2])]
                c2 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3])]
                c3 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4])]
                c = '%s%s%s%s' % (c0, c1, c2, c3)
                if label[0] == c:
                    correct = correct + 1
                total_size = total_size + 1
            print("accuracy =", correct/total_size)

    correct = 0
    total_size = 0
    for step, (img, label_oh, label) in enumerate(test_dl):
        img = Variable(img)
        pred = model(img)

        c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
        c1 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN * 2])]
        c2 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN * 2:ALL_CHAR_SET_LEN * 3])]
        c3 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN * 3:ALL_CHAR_SET_LEN * 4])]
        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print("original: ", label[0], " predict: ", c)
        if label[0] == c:
            correct = correct + 1
        total_size = total_size + 1

    print("accuracy =", correct / total_size)
    torch.save(model, "break_1.pt")