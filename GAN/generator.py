import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, in_c, out_c, n_res = 6, enc_kern = 3):
        """
            ADAPTED FROM JOHNSON ET AL
            in_c - number of input channels
            out_c - number of output channels
            n_res - number of resnet blocks between upsampling and downsampling
        """

        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, 7),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]

        model += [Downsample(64, 128, enc_kern, 2),
                 Downsample(128, 256, enc_kern, 2)]
        
        for i in range(n_res):
            model += [Residual(256, 3)]

        model += [Upsample(256, 128, enc_kern, 2),
                  Upsample(128, 64, enc_kern, 2)]
        
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7),
                  nn.BatchNorm2d(3),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print("in", x.shape)
        for i, module in enumerate(self.model):
            x = module(x)
            #print(i, x.shape)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_c, out_c, kern, stride = 2):
        super(Downsample, self).__init__()

        model = [#nn.ReflectionPad2d(int(kern/2)),
                 nn.Conv2d(in_c, out_c, kernel_size=kern, stride=stride, padding=1), 
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

        if kern == 4:
            out_pad = 0
        else:
            out_pad = 1
        
        model = [nn.ConvTranspose2d(in_c, out_c, kernel_size=kern, stride=stride, padding=1, output_padding=out_pad),
                 nn.BatchNorm2d(out_c),
                 nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    

if __name__ == "__main__":
    generator = Generator(3, 3)

    print(generator)

    random_tensor = torch.randn(16, 3, 128, 128)
    random_tensor = random_tensor
    
    result = generator.forward(random_tensor)

    print(result.shape)
