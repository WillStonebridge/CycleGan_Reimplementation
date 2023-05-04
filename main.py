import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.datasets import *
from GAN.CycleGAN import *
from visualize import *
import argparse
import time

#COMMAND LINE
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="test name")
ap.add_argument("-lr", "--lr", default=2e-4, help="learning rate")
ap.add_argument("-b", "--batch", default=1, help="batch size")
ap.add_argument("-e", "--epochs", default=150, help="epochs")
ap.add_argument("-ed", "--e_decay", default=100, help="What epoch does the learning rate begin decay at?")
ap.add_argument("-g", "--glayers", default=6, help="generator layers")
ap.add_argument("-d", "--dlayers", default=4, help="discriminator layers")
ap.add_argument("-adv", "--adv_coeff", default=1, help="adversarial coefficient")
ap.add_argument("-cyc", "--cyc_coeff", default=10, help="cycle coefficient")
ap.add_argument("-idt", "--idt_coeff", default=0.5, help="identity coefficient")
ap.add_argument("-x", "--setX", default="aerial", help="X dataset")
ap.add_argument("-y", "--setY", default="gmaps", help="Y dataset")
args = vars(ap.parse_args())

#SETTINGS
learning_rate = float(args["lr"])
batch_size = int(args["batch"])
epochs = int(args["epochs"])
e_decay = int(args["e_decay"])
g_layers = int(args["glayers"])
d_layers = int(args["dlayers"])
adv = float(args["adv_coeff"])
cyc = float(args["cyc_coeff"])
idt = float(args["idt_coeff"])
setX = args["setX"]
setY = args["setY"]
savefreq = 50 #how often is the model saved in minutes
name = args["name"]

if __name__ == "__main__":
    print(name)

    #DATASET
    dataset = Dataset(setX, setY)
    dataloader = dataset.get_loader(batch_size, shuffle=True)

    #MODEL, DEVICE
    device = torch.device("cuda")
    gan = cycleGAN(device, lr=learning_rate, epochs= epochs, e_decay=e_decay,
                   g_layers = g_layers, d_layers = d_layers,
                   coeff_adv = adv, coeff_forward=cyc, coeff_backward=cyc, coeff_idt = idt)
    gan.to(device)

    torch.cuda.empty_cache() #free some memory up
    insights = Insights(len(dataloader), epochs, name)
    insights.instantiateSaveFile(args)
    start_time = time.time()
    save_times = [(start_time + x*savefreq*60 + 60) for x in range(1000)]
    curr_save = 0 #what save is the program on
    final_saved = False

    print(sum(p.numel() for p in gan.parameters()))

    #TRAINING
    for epoch in range(epochs):
        for i, batch in enumerate(iter(dataloader)):
            gan.forward(batch)
            gan.optimize_D()
            gan.optimize_G()
            
            torch.cuda.empty_cache() #free some memory up

            if i % 100 == 0:
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






    