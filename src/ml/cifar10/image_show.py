import matplotlib.pyplot as plt
import numpy as np
import torchvision
from dataset import CIFAR10DataModule

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# batch_size = 32
# # get some random training images
# dm = CIFAR10DataModule(data_dir="./data", batch_size=32)
# dm.prepare_data()          # downloads
# dm.setup(stage=None)       # builds datasets

# train_loader = dm.train_dataloader()
# dataiter = iter(train_loader)
# images, labels = next(dataiter)


# # print labels
# classes = dm.trainset.classes
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# # show images
# imshow(torchvision.utils.make_grid(images))
