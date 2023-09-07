import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch.optim as optim
import torch.nn as nn
import re
from torch.utils.data import DataLoader
from datasets.datasets import *
from GAN.CycleGAN import *
from visualize import *
import argparse
import time

def getHyperParams(fpath):
    params = {}
    with open(f"{fpath}/about.txt", "r") as f:
        lines = f.readlines()[1:]
        pattern = r'^\s*Basics\s*-\s*(?P<lr>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*lr,\s*(?P<batches>\d+)\s*batches,\s*(?P<epochs>\d+)\s*epochs,\s*decay\s*at\s*(?P<decay>\d+)\s*$'
        match = re.match(pattern, lines[0])

        
        if match:
            params['lr'] = float(match.group('lr'))
            params['batch'] = int(match.group('batches'))
            params['epochs'] = int(match.group('epochs'))
            params['e_decay'] = int(match.group('decay'))
        else:
            print("ERROR READING FILE")

        pattern = r'\d+(?:\.\d+)?'
        match = re.findall(pattern, lines[1])
        params["adv_coeff"] = match[0]
        params["cyc_coeff"] = match[1]
        params["idt_coeff"] = match[2]

        architecture = re.findall(r"\d+", lines[2])
        params['glayers'] = int(architecture[0])
        params['dlayers'] = int(architecture[1])
        params['setX'] = lines[3].split(" ")[1][:-1]
        params['setY'] = lines[3].split(" ")[3]

        f.close()

    return params

def findStateDict(fpath):
    for dir in os.listdir(fpath):
        if os.path.isdir(f'{fpath}/{dir}'):
            for file in os.listdir(f'{fpath}/{dir}'):
                if file[-3:] == '.pt':
                    print(f'{fpath}/{dir}/{file}')
                    pattern = r"\d+"
                    matches = re.findall(pattern, dir)
                    curr_epoch = int(matches[0])
                    return curr_epoch, f'{fpath}/{dir}/{file}'


if __name__ == "__main__":
    res = 256

    #COMMAND LINE
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gan", required=True, help="gan path")
    ap.add_argument("-i", "--src", required=True, help="src path")
    ap.add_argument("-d", "--dst", required=True, help="dst path")
    args = vars(ap.parse_args())
    params = getHyperParams(args['gan'])
    curr_epoch, path_statedict = findStateDict(args['gan'])
    src = args['src']
    dst = args['dst']

    #SETTINGS
    learning_rate = float(params["lr"])
    batch_size = int(params["batch"])
    epochs = int(params["epochs"])
    e_decay = int(params["e_decay"])
    g_layers = int(params["glayers"])
    d_layers = int(params["dlayers"])
    adv = float(params["adv_coeff"])
    cyc = float(params["cyc_coeff"])
    idt = float(params["idt_coeff"])
    setX = params["setX"]
    setY = params["setY"]
    name = args["gan"][7:]
    
    device = torch.device('cpu')
    gan = cycleGAN(device, lr=learning_rate, epochs= epochs, e_decay=e_decay,
                   g_layers = g_layers, d_layers = d_layers,
                   coeff_adv = adv, coeff_forward=cyc, coeff_backward=cyc, coeff_idt = idt)
    gan.load_state_dict(torch.load(path_statedict))

    img = Image.open(src).convert("RGB")
    if img.size != (res, res):
            img = img.resize((res, res))
    transform = transforms.ToTensor()
    in_tensor = transform(img).unsqueeze(0) 

    result_tensor = gan.G_X(in_tensor)[0].detach()

    result_arr = result_tensor.permute(1, 2, 0).numpy()

    result_arr = np.clip(result_arr, 0, 1)

    plt.figure(1)
    plt.imsave(dst, result_arr)
    print(dst)





