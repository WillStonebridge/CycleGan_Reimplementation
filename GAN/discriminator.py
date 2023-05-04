import torch.nn as nn
import torch

class Discriminator(nn.Module):
    """
        Markovian Discriminator architecture adapted from Isola et Al
    """
    
    def __init__(self, in_c, layers=4, filters=64):

        super(Discriminator, self).__init__()

        sequence = [nn.Conv2d(in_c, filters, kernel_size=4, stride=2, padding=0), 
                    nn.LeakyReLU(0.2, True)]
        
        f_mult = 1
        p_f_mult = 1
        for layer in range(layers-1):
            p_f_mult = f_mult
            f_mult *= 2
            if not layer == layers-1:
                sequence += [nn.Conv2d(filters * p_f_mult, filters * f_mult, 4, 2)]
            else:
                sequence += [nn.Conv2d(filters * p_f_mult, filters * f_mult, 4, 2)]

            sequence += [nn.BatchNorm2d(filters*f_mult),
                         nn.LeakyReLU(0.2, True)]
            
        
        sequence += [nn.Conv2d(filters * f_mult, 1, 4, 1),
                     nn.Sigmoid()]

        
        self.model = nn.Sequential(*sequence)


    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x





if __name__ == "__main__":
    d = Discriminator(3, 4, 64)
    random_tensor = torch.randn(16, 3, 256, 256)
    result = d.forward(random_tensor)

    print(result.shape)
    print(result)