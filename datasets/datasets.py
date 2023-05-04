from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import random
import scipy.io as sio
from PIL import Image
import numpy as np

class Dataset():
    def __init__(self, setA, setB, resolution=256):
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

        imgA = jitter(imgA)
        imgB = jitter(imgB)

        tensorA = self.transform(imgA) #turn each image into a tensor
        tensorB = self.transform(imgB)
        
        return tensorA, tensorB


    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)
    

def random_crop(img): #crop an image into a random section of itself, while preserving its size
    width, height = img.size

    target_size = (int(width*1.1), int(height*1.1)) #enlarge image so it can be cropped to original size
    img = img.resize(target_size, resample=Image.NEAREST) 
    
    x = random.randint(0, target_size[0] - width) #start crop x
    y = random.randint(0, target_size[1] - height) #start crop y
    
    return img.crop((x, y, x + width, y + height))

def jitter(img):
    img = random_crop(img) #randomly crop the image

    if random.random() > 0.5: #50% chance the image is flipped
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img




if __name__ == "__main__":
    # get the current directory path
    current_dir = os.getcwd()

    # get the parent directory path
    parent_dir = os.path.dirname(current_dir)

    # change the current directory to the parent directory
    os.chdir(parent_dir)


    dataset = Dataset('water', 'wine').get_loader(1, True)
    for i, tensors in enumerate(dataset):
        if (tensors[0].shape[1] == 1):
            print("A")
            break
        if (tensors[1].shape[1] == 1):
            print("B")
            break
        else:
            print(tensors[0].shape)
            print(tensors[1].shape)


