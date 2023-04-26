import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, in_c, out_c, n_res = 3):
        """
            ADAPTED FROM JOHNSON ET AL
            in_c - number of input channels
            out_c - number of output channels
            n_res - number of resnet blocks between upsampling and downsampling
        """

        super(Generator, self).__init__()

        model = [Downsample(3, 32, 9,),
                 Downsample(32, 64, 3),
                 Downsample(64, 128, 3)]
        
        for i in range(n_res):
            model += [Residual(128, 3)]

        model += [Upsample(128, 64, 3, 1),
                  Upsample(64, 32, 3, 1)]
        
        model += [nn.Conv2d(32, 3, kernel_size=9),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        for module in self.model:
            x = module(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_c, out_c, kern):
        super(Downsample, self).__init__()

        model = [nn.ReflectionPad2d(int(kern/2)),
                 nn.Conv2d(in_c, out_c, kernel_size=kern), 
                 nn.BatchNorm2d(out_c),
                 nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Residual(nn.Module):

    def __init__(self, ch, kern):
        super(Residual, self).__init__()
        """
            A convolutional sequence in which the channels are reflected, 
            convolved and then normalized twice with a ReLU inbetween
        """
        Refl_Conv_Norm = [nn.ReflectionPad2d(int(kern/2)),
                          nn.Conv2d(ch, ch, kernel_size=kern),
                          nn.BatchNorm2d(ch)]

        sequence = Refl_Conv_Norm
        sequence += [nn.ReLU(True)]
        sequence += Refl_Conv_Norm

        self.model = nn.Sequential(*sequence)
        

    def forward(self, x):
        """
            Forward of the resblock 
        """
        return self.model(x) + x


class Upsample(nn.Module):

    def __init__(self, in_c, out_c, kern, stride):
        super(Upsample, self).__init__()
        
        model = [nn.ReflectionPad2d(int(kern/2)),
                 nn.ConvTranspose2d(in_c, out_c, kernel_size=kern, stride=stride),
                 nn.BatchNorm2d(out_c),
                 nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    

if __name__ == "__main__":
    generator = Generator(3, 3)
    generator.to(torch.device("cuda"))

    print(generator)

    random_tensor = torch.randn(16, 3, 32, 32)
    random_tensor = random_tensor.to(torch.device("cuda"))
    
    result = generator.forward(random_tensor)


    print(result.shape)
    print(result)
