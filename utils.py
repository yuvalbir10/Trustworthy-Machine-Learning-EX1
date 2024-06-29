import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    accuracy = 0
    total_samples = 0
    model.eval() # TODO: this is already done in the main_a.py, consider removing it
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy /= total_samples
    return accuracy


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    adversarial_samples = []
    original_images = []
    # the original labels for untargeted attacks
    true_labels = []
    # the target desired labels for targeted attacks
    target_labels = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        original_images.append(inputs)
        if targeted:
            rand_targeted_labels = (labels + torch.randint(1, n_classes, labels.size())) % n_classes
            target_labels.append(rand_targeted_labels)
            rand_targeted_labels = rand_targeted_labels.to(device)
        else:
            true_labels.append(labels)
            labels = labels.to(device)

        perturbed_inputs = attack.execute(inputs, rand_targeted_labels if targeted else labels, targeted)
        adversarial_samples.append(perturbed_inputs)
        
    adversarial_samples = torch.cat(adversarial_samples)
    if targeted:
        target_labels = torch.cat(target_labels)
    else:
        true_labels = torch.cat(true_labels)

    original_images = torch.cat(original_images)

    assert torch.all((adversarial_samples >= 0) & (adversarial_samples <= 1)), "adversarial_samples are not within the range of 0 to 1"
    assert torch.all(adversarial_samples >= original_images - attack.eps) and torch.all(adversarial_samples <= original_images + attack.eps), "Adversarial samples are not within the epsilon-ball of the original samples"

    return adversarial_samples, (target_labels if targeted else true_labels)


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    adversarial_samples = []
    true_labels = []
    target_labels = []
    num_queries = []
    original_images = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        original_images.append(inputs)
        if targeted:
            rand_targeted_labels = (labels + torch.randint(1, n_classes, labels.size())) % n_classes
            target_labels.append(rand_targeted_labels)
            rand_targeted_labels = rand_targeted_labels.to(device)
        else:
            true_labels.append(labels)
            labels = labels.to(device)

        perturbed_inputs, queries = attack.execute(inputs, rand_targeted_labels if targeted else labels, targeted)
        adversarial_samples.append(perturbed_inputs)
        num_queries.append(queries)
        
    adversarial_samples = torch.cat(adversarial_samples)
    num_queries = torch.cat(num_queries)
    original_images = torch.cat(original_images)

    if targeted:
        target_labels = torch.cat(target_labels)
    else:
        true_labels = torch.cat(true_labels)

    assert torch.all((adversarial_samples >= 0) & (adversarial_samples <= 1)), "adversarial_samples are not within the range of 0 to 1"
    assert torch.all(adversarial_samples >= original_images - attack.eps) and torch.all(adversarial_samples <= original_images + attack.eps), "Adversarial samples are not within the epsilon-ball of the original samples"

    return adversarial_samples, (target_labels if targeted else true_labels), num_queries


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    success = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_adv), batch_size):
            inputs = x_adv[i:i+batch_size].to(device)
            labels = y[i:i+batch_size].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if targeted:
                success += (predicted == labels).sum().item()
            else:
                success += (predicted != labels).sum().item()
            total_samples += labels.size(0)
    success /= total_samples
    return success



def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass  # FILL ME


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass  # FILL ME


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass  # FILL ME
