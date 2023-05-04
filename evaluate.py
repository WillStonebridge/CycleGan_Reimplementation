import os 
import torch
import re 
import random
import numpy as np
import argparse
from GAN.CycleGAN import cycleGAN
from datasets.datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_hyperparams(fpath):
    params = {}
    with open(f"{fpath}/about.txt", "r") as f:
        lines = f.readlines()[1:]
        architecture = re.findall(r"\d+", lines[2])
        params['g'] = int(architecture[0])
        params['d'] = int(architecture[1])
        params['x'] = lines[3].split(" ")[1][:-1]
        params['y'] = lines[3].split(" ")[3]
        f.close()

    return params


def most_recent_path(fpath):
    dirs = []
    for entry in os.scandir(fpath):
        if entry.is_dir():
            dirs.append(entry.name)

    print(dirs)
    return f"{dirs[-1]}/model.pt"
    

def eval_model(gan, x, y, save_dir, num_imgs):
    #GET IMAGES
    dataloader =  Dataset(x, y).get_loader(1, shuffle=True)
    x = []
    y = []
    gx = []
    gy = []
    for i, batch in enumerate(iter(dataloader)):
        if i == num_imgs:
            break

        
        X = batch[0].to(gan.device)
        Y = batch[1].to(gan.device)
        Gx = gan.G_X(X)[0].detach().cpu()
        Gy = gan.G_Y(Y)[0].detach().cpu()
        X = X[0].cpu()
        Y = Y[0].cpu()


        X = X.permute(1, 2, 0).numpy()
        Y = Y.permute(1, 2, 0).numpy()
        Gx = Gx.permute(1, 2, 0).numpy()
        Gy = Gy.permute(1, 2, 0).numpy()
        Gx = np.clip(Gx, 0, 1)
        Gy = np.clip(Gy, 0, 1)

        x.append(X)
        y.append(Y)
        gx.append(Gx)
        gy.append(Gy)


    #PLOT 
    figure_id = random.random() * 10000
    gs = gridspec.GridSpec(num_imgs, 4) # subplot_params={'hspace': 0.5, 'wspace': 0.2}
    fig = plt.figure(figure_id, figsize=(12, 25), dpi=190, clear=True) #figsize=(8, 4), dpi=190
    x_plt = []
    y_plt = []
    gx_plt = []
    gy_plt = []
    for i in range(num_imgs):
        x_plt.append(fig.add_subplot(gs[i, 0]))
        gx_plt.append(fig.add_subplot(gs[i, 1]))
        y_plt.append(fig.add_subplot(gs[i, 2]))
        gy_plt.append(fig.add_subplot(gs[i, 3]))

    for i in range(num_imgs):
        x_plt[i].imshow(x[i])
        x_plt[i].axis('off')
        y_plt[i].imshow(y[i])
        y_plt[i].axis('off')
        gx_plt[i].imshow(gx[i])
        gx_plt[i].axis('off')
        gy_plt[i].imshow(gy[i])
        gy_plt[i].axis('off')

    fig.savefig(f"{save_dir}/samples")
    print(f"{save_dir}/samples.png")


def eval_saved_model(fpath, num_imgs):
    params = get_hyperparams(f"../{fpath}")

    #LOAD GAN
    gan = cycleGAN(torch.device("cpu"), 5e-4, g_layers=params['g'], d_layers=params['d'])
    gan.load_state_dict(torch.load(f"{fpath}/model.pt"))

    eval_model(gan, params['x'], params['y'], )

    #GET IMAGES
    dataloader =  Dataset(params['x'], params['y']).get_loader(1, shuffle=True)
    x = []
    y = []
    gx = []
    gy = []
    for i, batch in enumerate(iter(dataloader)):
        if i == num_imgs:
            break


        X = gan.real_X[0, :, :, :].cpu()
        Y = gan.real_Y[0, :, :, :].cpu()
        Gy = gan.generated_X[0, :, :, :].detach().cpu()
        Gx = gan.generated_Y[0, :, :, :].detach().cpu()

        X = X.permute(1, 2, 0).numpy()
        Y = Y.permute(1, 2, 0).numpy()
        Gx = Gx.permute(1, 2, 0).numpy()
        Gy = Gy.permute(1, 2, 0).numpy()
        Gx = np.clip(Gx, 0, 1)
        Gy = np.clip(Gy, 0, 1)
        
        X = batch[0]
        Y = batch[1]
        Gx = gan.G_X(X)[0].detach()
        Gy = gan.G_Y(Y)[0].detach()
        X = X[0]
        Y = Y[0]


        X = X.permute(1, 2, 0).numpy()
        Y = Y.permute(1, 2, 0).numpy()
        Gx = Gx.permute(1, 2, 0).numpy()
        Gy = Gy.permute(1, 2, 0).numpy()
        Gx = np.clip(Gx, 0, 1)
        Gy = np.clip(Gy, 0, 1)

        x.append(X)
        y.append(Y)
        gx.append(Gx)
        gy.append(Gy)


    #PLOT 
    gs = gridspec.GridSpec(num_imgs, 4) # subplot_params={'hspace': 0.5, 'wspace': 0.2}
    fig = plt.figure(1, figsize=(12, 25), dpi=190) #figsize=(8, 4), dpi=190
    x_plt = []
    y_plt = []
    gx_plt = []
    gy_plt = []
    for i in range(num_imgs):
        x_plt.append(fig.add_subplot(gs[i, 0]))
        gx_plt.append(fig.add_subplot(gs[i, 1]))
        y_plt.append(fig.add_subplot(gs[i, 2]))
        gy_plt.append(fig.add_subplot(gs[i, 3]))

    for i in range(num_imgs):
        x_plt[i].imshow(x[i])
        x_plt[i].axis('off')
        y_plt[i].imshow(y[i])
        y_plt[i].axis('off')
        gx_plt[i].imshow(gx[i])
        gx_plt[i].axis('off')
        gy_plt[i].imshow(gy[i])
        gy_plt[i].axis('off')

    plt.savefig(f"{fpath}/samples")
    print(f"{fpath}/samples.png")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="model path")
    ap.add_argument("-n", "--n_imgs", default=8, help="number of images")
    args = vars(ap.parse_args())

    eval_saved_model(args['path'], int(args['n_imgs']))
