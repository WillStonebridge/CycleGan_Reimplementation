import torch
from torch import optim

from GAN.generator import Generator
from GAN.discriminator import *
from GAN.FakeBuffer import *
import itertools

"""from generator import Generator
from discriminator import *"""

class cycleGAN(nn.Module):

    def __init__(self, device, lr, g_layers = 6, d_layers = 4, g_filters = 64, d_filters = 64, coeff_adv = 1, coeff_forward = 1, coeff_backward = 1, coeff_idt = 1, epochs=200, e_decay = 100):
        super(cycleGAN, self).__init__()

        #NETWORKS
        self.G_X = Generator(3, 3, g_layers, enc_kern=3)
        self.G_Y = Generator(3, 3, g_layers, enc_kern=3)
        self.D_X = Discriminator(3, d_layers, d_filters)
        self.D_Y = Discriminator(3, d_layers, d_filters)

        #BUFFERS
        self.buffer_Gx = FakeBuffer()
        self.buffer_Gy = FakeBuffer()

        #LOSSES and Coefficients
        #note: gan loss is defined externally
        self.l1loss = nn.L1Loss()
        self.coeff_adv = coeff_adv
        self.coeff_forw = coeff_forward
        self.coeff_back = coeff_backward
        self.coeff_idt = coeff_idt
        self.coeff_Dx_real = 1
        self.coeff_Dx_fake = 1
        self.coeff_Dy_real = 1
        self.coeff_Dy_fake = 1
        self.loss_G = None

        #OPTIMIZERS
        self.opt_G = optim.Adam(itertools.chain(self.G_X.parameters(), self.G_Y.parameters()), lr=lr)
        self.opt_D = optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=lr)

        #SCHEDULERS
        def lin_decay(epoch):
            decay = 1.0
            if epoch > e_decay: #only start decaying at epoch e_decay
                decay = 1.0 - ((epochs - e_decay) - (epoch - e_decay)) / (epochs - e_decay)
            return decay
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lin_decay)
        self.scheduler_D = optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda=lin_decay)

        self.device = device

    def forward(self, batch):
        self.real_X = batch[0].to(self.device)
        self.real_Y = batch[1].to(self.device)

        """print("\n\nchecking within forward:\n")
        print("input ",self.real_A.device)
        print("params ", next(self.G_A.parameters()).device)
        for name, param in self.G_A.named_parameters():
            if param.requires_grad:
                print(name, param.data.device)"""

        self.generated_Y = self.G_X(self.real_X)
        self.generated_X = self.G_Y(self.real_Y)
        self.recovered_Y = self.G_X(self.generated_X)
        self.recovered_X = self.G_Y(self.generated_Y)


    def backward_D(self):
        """
            Calculates the loss of both discriminators on real and fake samples, calculates the gradient of each
        """

        #Mix in new generated data with old generated data
        generated_X = self.buffer_Gx.get_fakes(self.generated_X)
        generated_Y = self.buffer_Gy.get_fakes(self.generated_Y)

        self.loss_Dx_find_real = getGANLoss(self.D_X(self.real_X), True) #how bad is the discriminator at identifying True X
        self.loss_Dx_find_fake = getGANLoss(self.D_X(generated_X), False) #how bad is the discriminator at identifying Fake X
        self.loss_Dx = (self.coeff_Dx_real*self.loss_Dx_find_real + self.coeff_Dx_fake*self.loss_Dx_find_fake) #total loss is the combination of both divided by 2

        self.loss_Dy_find_real = getGANLoss(self.D_Y(self.real_Y), True) #how bad is the Dy at identifying True Y
        self.loss_Dy_find_fake = getGANLoss(self.D_Y(generated_Y), False) #how bad is the Dy at identifying Fake Y
        self.loss_Dy = (self.coeff_Dy_real*self.loss_Dy_find_real + self.coeff_Dx_real*self.loss_Dy_find_fake) #total loss is the combination of both divided by 2

        self.adjust_D_coeffs() #if either discriminator is unstable, alter it's loss coefficients

        #calculate the gradients of both losses
        self.loss_Dx.backward()
        self.loss_Dy.backward()

    def adjust_D_coeffs(self):
        instability_bound = 0.8 #the bound at which we determine the discriminator loss to be unstable
        stability_bound = 0.7
        favor = 0.2
        discourage = 1.8
        if self.loss_Dx_find_real > instability_bound:
            self.coeff_Dx_real = discourage #favor classifying fake
            self.coeff_Dx_fake = favor
        elif self.loss_Dx_find_fake > instability_bound: 
            self.coeff_Dx_real = favor #favor classifying real
            self.coeff_Dx_fake = discourage
        elif self.loss_Dx_find_real < stability_bound and self.loss_Dx_find_fake < stability_bound: #Dx is stable
            self.coeff_Dx_real = 1 #favor neither
            self.coeff_Dx_fake = 1
        

        if self.loss_Dy_find_real > instability_bound:
            self.coeff_Dy_real = discourage #favor classifying fake
            self.coeff_Dy_fake = favor
        elif self.loss_Dy_find_fake > instability_bound: 
            self.coeff_Dy_real = favor #favor classifying real
            self.coeff_Dy_fake = discourage
        elif self.loss_Dy_find_real < stability_bound and self.loss_Dy_find_fake < stability_bound:
            self.coeff_Dy_real = 1 #favor neither
            self.coeff_Dy_fake = 1
    
    def backward_G(self):
        #GAN LOSS
        self.Gx_loss = getGANLoss(self.D_X(self.generated_X), True) #how well can Gx(X) trick Dy
        self.Gy_loss = getGANLoss(self.D_Y(self.generated_Y), True) #how well can Gy(Y) trick Dx
        self.adv_loss = self.coeff_adv * (self.Gx_loss + self.Gy_loss)

        #CYCLE CONSISTENCY LOSS: L1 between the original and Gy(Gx(A))
        self.cyc_X_loss = self.l1loss(self.real_X, self.recovered_X)
        self.cyc_Y_loss = self.l1loss(self.real_Y, self.recovered_Y)
        self.cyc_loss = self.coeff_forw * self.cyc_X_loss + self.coeff_back * self.cyc_Y_loss

        #IDENTITY LOSS
        self.idt_y_loss = self.l1loss(self.G_X(self.real_Y), self.real_Y)
        self.idt_x_loss = self.l1loss(self.G_Y(self.real_X), self.real_X)
        self.idt_loss = self.coeff_idt * (self.idt_x_loss + self.idt_y_loss)

        self.loss_G = self.Gx_loss + self.Gy_loss + self.cyc_loss + self.idt_loss

        self.loss_G.backward()


    def optimize_G(self):
        #Turn off gradient calculations for discriminators
        self.grad_active(self.D_X, False)
        self.grad_active(self.D_Y, False)

        self.opt_G.zero_grad() #the gradient is set to zero
        self.backward_G() #get the gradient
        #if self.loss_Dx_find_fake < 0.4 and self.loss_Dx_find_real < 0.4: #if Dx is stable
        self.opt_G.step() #update parameters

        #Turn back on gradient calculations for discriminators
        self.grad_active(self.D_X, True)
        self.grad_active(self.D_Y, True)


    def optimize_D(self):
        self.backward_D()
        self.opt_D.step()

    def step_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def grad_active(self, net, On):
        for parameter in net.parameters(): 
            parameter.requires_grad = On

    def display_loss(self):
        print(f"GENERATOR LOSS: G:{self.loss_G}")
        print(f"CYCLE LOSS - Forward:{self.cyc_X_loss}, Backward:{self.cyc_Y_loss}")
        print(f"GAN LOSS - Gx:{self.Ga_loss}, Gy:{self.Gb_loss}")
        print(f"DISCRIMINATOR LOSS - Dx:{self.loss_Dx}, Dy:{self.loss_Dy}")



mse_loss = nn.MSELoss() #defined outside of the function to avoid initialization on every call
mse_loss.to(torch.device("cuda"))

def getGANLoss(D_output, output_is_real):
    """
        Takes the output of a discriminator on a image and a boolean representing whether the original image
        was fake or not
    
        returns the MSE of the discriminator output and expected discriminator output
    """

    if output_is_real:
        expectation = torch.ones(D_output.shape)
    else:
        expectation = torch.zeros(D_output.shape)

    expectation = expectation.to(torch.device('cuda'))

    return mse_loss(D_output, expectation)


if __name__ == "__main__":
    device = torch.device("cuda")
    gan = cycleGAN(device, lr=2e-4)

    ones = torch.ones((32, 3, 64, 64))

    gan.forward([ones, ones])