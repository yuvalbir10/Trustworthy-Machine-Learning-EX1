import utils
import consts
import models
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f'Model accuracy before flipping: {acc_orig:0.4f}')

# layers whose weights will be flipped
layers = {'conv1': model.conv1,
          'conv2': model.conv2,
          'fc1': model.fc1,
          'fc2': model.fc2,
          'fc3': model.fc3}

# flip bits at random and measure impact on accuracy (via RAD)
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)])  # will contain a list of RADs for each index of bit flipped
RADs_all = []  # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False
        for _ in range(consts.BF_PER_LAYER):
            # randomly pick a weight
            weight_idx = random.randint(0, len(W.view(-1)) - 1)
            weight = W.view(-1)[weight_idx]
            # randomly pick a bit to flip
            bit_idx = random.randint(0, 31)
            # flip the bit
            flipped_weight = int(weight) ^ (1 << bit_idx)
            # update the weight with the flipped bit
            W.view(-1)[weight_idx] = flipped_weight
            # compute the accuracy after bit flipping
            acc_bf = utils.compute_accuracy(model, data_loader, device)
            # compute the RAD
            rad = (acc_orig - acc_bf) / acc_orig
            # restore the weight
            W.view(-1)[weight_idx] = weight
            # append the RAD to the corresponding bit-flip index
            RADs_bf_idx[bit_idx].append(rad)
            # append the RAD to the overall RADs
            RADs_all.append(rad)

# Max and % RAD>10%
RADs_all = np.array(RADs_all)
print(f'Total # weights flipped: {len(RADs_all)}')
print(f'Max RAD: {np.max(RADs_all):0.4f}')
print(f'RAD>10%: {np.sum(RADs_all > 0.1) / RADs_all.size:0.4f}')

# boxplots: bit-flip index vs. RAD
plt.figure()
plt.boxplot(RADs_bf_idx.values())
plt.xlabel('Bit-Flip Index')
plt.ylabel('Relative Accuracy Drop (RAD)')
plt.title('Bit-Flip Index vs. RAD')
plt.savefig('bf_idx-vs-RAD.jpg')
