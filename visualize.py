from PIL import Image
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GAN.CycleGAN import *
from evaluate import *
import torch
import shutil
import numpy as np
import os


class Insights():

    def __init__(self, batches, epochs, name = None):
        figure_id = random.random() * 10000

        if name != None:
            self.savePath = f"models/{name}"
        else:
            self.savePath = None

        gs = gridspec.GridSpec(2, 6, hspace = 1.4, wspace = 0.5) # subplot_params={'hspace': 0.5, 'wspace': 0.2}
        self.fig = plt.figure(figure_id, figsize=(11, 4), dpi=190)
        self.graph_gx = self.fig.add_subplot(gs[0, :2]) #graph is 4 times as large as the images
        self.graph_gy = self.fig.add_subplot(gs[0, 2:4])
        self.graph_dx = self.fig.add_subplot(gs[1, :2])
        self.graph_dy = self.fig.add_subplot(gs[1, 2:4])
        self.x_plt = self.fig.add_subplot(gs[0, 4]) #each image is equally as large
        self.y_plt = self.fig.add_subplot(gs[1, 4])
        self.gx_plt = self.fig.add_subplot(gs[0, 5])
        self.gy_plt = self.fig.add_subplot(gs[1, 5])

        #for the gx graph
        self.gx_adv_loss = []
        self.gx_cyc_loss = []
        self.gx_idt_loss = []

        #for the gy graph
        self.gy_adv_loss = []
        self.gy_cyc_loss = []
        self.gy_idt_loss = []
        
        #for the dx graph
        self.dx_loss_fake = []
        self.dx_loss_real = []

        #for the dy graph
        self.dy_loss_real = []
        self.dy_loss_fake = []
        
        self.x_axis = []
        self.x = 0
        self.batches = batches
        self.epochs = epochs


    def get_sample(self, gan):
        #get 1 sample 
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
  
        self.gx_adv_loss.append(float(gan.Gx_loss.cpu()))
        self.gx_cyc_loss.append(float(gan.cyc_X_loss.cpu()))
        self.gx_idt_loss.append(float(gan.idt_x_loss.cpu()))

        self.gy_adv_loss.append(float(gan.Gy_loss.cpu()))
        self.gy_cyc_loss.append(float(gan.cyc_Y_loss.cpu()))
        self.gy_idt_loss.append(float(gan.idt_y_loss.cpu()))

        self.dy_loss_real.append(float(gan.loss_Dy_find_real.cpu()))
        self.dx_loss_real.append(float(gan.loss_Dx_find_real.cpu()))

        self.dy_loss_fake.append(float(gan.loss_Dy_find_fake.cpu()))
        self.dx_loss_fake.append(float(gan.loss_Dx_find_fake.cpu()))

        self.x_axis.append(self.x)
        self.x += 1

        self.X = X
        self.Y = Y
        self.Gx = Gx
        self.Gy = Gy


    def display_insights(self, gan, batch, epoch):
        plt.ion()

        self.graph_gx.clear() #clear the prior subplots
        self.graph_gy.clear()
        self.graph_dy.clear()
        self.graph_dx.clear()
        self.x_plt.clear()
        self.y_plt.clear()
        self.gx_plt.clear()
        self.gy_plt.clear()

        #set up images
        self.x_plt.imshow(self.X)
        self.x_plt.set_title("X")
        self.x_plt.axis("off")
        self.y_plt.imshow(self.Y)
        self.y_plt.set_title("Y")
        self.y_plt.axis("off")
        self.gx_plt.imshow(self.Gx)
        self.gx_plt.set_title("Gx")
        self.gx_plt.axis("off")
        self.gy_plt.imshow(self.Gy)
        self.gy_plt.set_title("Gy")
        self.gy_plt.axis("off")

        #set up graph gx
        self.graph_gx.set_title("Gx: X->Y")
        self.graph_gx.set_ylabel("Loss")
        self.graph_gx.set_xlabel("Updates")
        self.graph_gx.plot(self.x_axis, self.gx_adv_loss, label="Adv")
        self.graph_gx.plot(self.x_axis, self.gx_cyc_loss, label="Cyc")
        self.graph_gx.plot(self.x_axis, self.gx_idt_loss, label="Idt")
        self.graph_gx.legend(fontsize='x-small')

        #set up graph gy
        self.graph_gy.set_title("Gy: Y->X")
        self.graph_gy.set_ylabel("Loss")
        self.graph_gy.set_xlabel("Updates")
        self.graph_gy.plot(self.x_axis, self.gy_adv_loss, label="Adv")
        self.graph_gy.plot(self.x_axis, self.gy_cyc_loss, label="Cyc")
        self.graph_gy.plot(self.x_axis, self.gy_idt_loss, label="Idt")
        self.graph_gy.legend(fontsize='x-small')

        #set up graph dx
        self.graph_dx.set_title("Dx: Gx(x) vs Y")
        self.graph_dx.set_ylabel("Loss")
        self.graph_dx.set_xlabel("Updates")
        self.graph_dx.plot(self.x_axis, self.dx_loss_real, label="Real")
        self.graph_dx.plot(self.x_axis, self.dx_loss_fake, label="Fake")
        self.graph_dx.legend(fontsize='x-small')

        #set up graph dy
        self.graph_dy.set_title("Dx: Gy(y) vs X")
        self.graph_dy.set_ylabel("Loss")
        self.graph_dy.set_xlabel("Updates")
        self.graph_dy.plot(self.x_axis, self.dy_loss_real, label="Real")
        self.graph_dy.plot(self.x_axis, self.dy_loss_fake, label="Fake")
        self.graph_dy.legend(fontsize='x-small')

        #display
        self.fig.suptitle(f"Insights - batch {batch}/{self.batches} of epoch {epoch+1}/{self.epochs}", fontsize=12)
        self.fig.canvas.draw()
        plt.pause(0.001)
        plt.show()

        plt.ioff()


    def instantiateSaveFile(self, args):
        if self.savePath != None:
            if os.path.exists(self.savePath): #delete the previous directory if there is already one named
                shutil.rmtree(self.savePath)
                print(f"deleted {self.savePath}")
            os.mkdir(self.savePath)
            print(f"created {self.savePath}")

            with open(f'{self.savePath}/about.txt', 'w') as f:
                f.write(f"NAME: {self.savePath[7:]}\n")
                f.write(f"Basics - {args['lr']} lr, {args['batch']} batches, {args['epochs']} epochs, decay at {args['e_decay']}\n")
                f.write(f"Coefficients - {args['adv_coeff']} adv, {args['cyc_coeff']} cyc, {args['idt_coeff']} idt\n")
                f.write(f"Architectures - {args['glayers']} glayers, {args['dlayers']} dlayers\n")
                f.write(f"X: {args['setX']}, Y: {args['setY']}")
                f.close()

    def save_figures(self, epoch, batch, gan, x, y):
        save_dir = f"{self.savePath}/e{epoch}b{batch}"
        os.mkdir(save_dir)

        self.fig.savefig(f"{save_dir}/insights.jpg")
        eval_model(gan, x, y, save_dir, 8)

        return save_dir
        


    def save_model(self, epoch, batch, gan , x, y):
        save_dir = self.save_figures(epoch, batch, gan, x, y)
        torch.save(gan.state_dict(), f"{save_dir}/model.pt")

if __name__ == "__main__":
    print("hello")