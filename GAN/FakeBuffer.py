import random
import torch

def split_batch(batch):
    images = []
    batch_size = batch.shape[0]
    for i in range(batch_size):
        images += [batch[i, :, :, :]]

    return images

def list_to_gpu_tensor(list):
    tensor_cpu = torch.stack(list, dim=0)
    tensor_gpu = tensor_cpu.to(torch.device("cuda"))
    
    return tensor_gpu

class FakeBuffer():
    """
        Training strategy adapted from shrivista et al
    
        A buffer that returns a generated batch which is 50% new and 50% old images
    """

    def __init__(self):
        self.capacity = 100
        self.buffer = []

    def select_old_new(self, fake, append_if_new):
        rv = random.random()
        if rv > 0.5: #return the new tensor
            if append_if_new:
                self.buffer.append(fake)
            return fake
        else: #return an old tensor, add the new one to the buffer
            ridx = int(random.random() * len(self.buffer)) #choose a random index
            old_fake = self.buffer[ridx] #get an old image
            self.buffer[ridx] = fake #replace the old image on the buffer with the new one
            return old_fake
    
    def get_fakes(self, fakes):
        fakes = split_batch(fakes.detach().cpu())
        if len(self.buffer) == 0:
            self.buffer += fakes #add a version of fakes in list form, keep them on the cpu
            return list_to_gpu_tensor(fakes)
        else:
            ret = []
            for fake in fakes:
                if len(self.buffer) < self.capacity:
                    ret.append(self.select_old_new(fake, True))
                else:
                    ret.append(self.select_old_new(fake, False))
            
            return list_to_gpu_tensor(ret)

            
if __name__ == "__main__":
    buf = FakeBuffer()

    for i in range(1000):
        batch = torch.randn(4, 1, 4, 4)

        buf.get_fakes(batch)

    print(buf.get_fakes(torch.ones(4, 1, 4, 4)))

