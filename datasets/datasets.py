from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import scipy.io as sio
from PIL import Image
import numpy as np

class Dataset():
    def __init__(self, setA, setB, resolution=128):
        self.dir_A_train = f"datasets/{setA}/train"
        self.dir_B_train = f"datasets/{setB}/train"
        self.dir_A_test = f"datasets/{setA}/test"
        self.dir_B_test = f"datasets/{setB}/test" 

        self.A_train_files = os.listdir(self.dir_A_train)
        self.B_train_files = os.listdir(self.dir_B_train)

        #The shortest dataset between A and B determines the length of the dataloader
        if len(self.A_train_files) < len(self.B_train_files):
            self.train_len = len(self.A_train_files)
        else:
            self.train_len = len(self.B_train_files)

        self.transform = transforms.ToTensor()
        self.res = resolution


    def __len__(self):
        return self.train_len


    def __getitem__(self, idx):
        pathA = f"{self.dir_A_train}/{self.A_train_files[idx]}" #get the path of each image
        pathB = f"{self.dir_B_train}/{self.B_train_files[idx]}"

        imgA = Image.open(pathA) #get PIL images of each
        imgB = Image.open(pathB)

        res = self.res
        if imgA.size != (res, res):
            imgA = imgA.resize((res, res))
        if imgB.size != (res,res):
            imgB = imgB.resize((res, res))

        tensorA = self.transform(imgA) #turn each image into a tensor
        tensorB = self.transform(imgB)
        
        return tensorA, tensorB


    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)


if __name__ == "__main__":
    dataset = Dataset('monet', 'photo')
    tensors = dataset.__getitem__(0)
    print(tensors[0])
    print(tensors[1])

