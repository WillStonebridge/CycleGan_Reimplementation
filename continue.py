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


#COMMAND LINE
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="test name")
args = vars(ap.parse_args())
params = getHyperParams(args['name'])
curr_epoch, path_statedict = findStateDict(args['name'])

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
savefreq = 45 #how often is the model saved in minutes
name = args["name"][7:]

if __name__ == "__main__":

    #DATASET
    dataset = Dataset(setX, setY)
    dataloader = dataset.get_loader(batch_size, shuffle=True)

    #MODEL, DEVICE
    device = torch.device("cuda")
    gan = cycleGAN(device, lr=learning_rate, epochs= epochs, e_decay=e_decay,
                   g_layers = g_layers, d_layers = d_layers,
                   coeff_adv = adv, coeff_forward=cyc, coeff_backward=cyc, coeff_idt = idt)
    gan.load_state_dict(torch.load(path_statedict))
    gan.to(device)

    torch.cuda.empty_cache() #free some memory up
    insights = Insights(len(dataloader), epochs, name)
    #insights.instantiateSaveFile(args)
    start_time = time.time()
    save_times = [(start_time + x*savefreq*60 + 60) for x in range(1000)]
    curr_save = 0 #what save is the program on
    final_saved = False

    #TRAINING
    for epoch in range(curr_epoch, epochs):
        for i, batch in enumerate(iter(dataloader)):
            gan.forward(batch)
            gan.optimize_D()
            gan.optimize_G()
            
            torch.cuda.empty_cache() #free some memory up

            if i % 25 == 0:
                insights.get_sample(gan)

                if save_times[curr_save] < time.time(): #save figures every checkpoint
                    print("Checkpoint")
                    curr_save += 1
                    insights.display_insights(gan, i, epoch)
                    insights.save_figures(epoch, i, gan, setX, setY)

                if not final_saved and time.time() - start_time > 60*60*4 - 10*60: #save the model when the time limit is near
                    print("Final Save")
                    insights.save_model(epoch, i, gan, setX, setY)
                    final_saved = True

                torch.cuda.empty_cache()

        gan.step_lr() #adjust the learning rate every epoch

    insights.save_model(epoch, i, gan, setX, setY) #save the model if training finishes before the time limit






    