import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image # tensor to pil_image
import torch
from dataset import MaskLabels, GenderLabels, AgeLabels

def show_images(dataset, idxs, rows=1, cols=7):
    plt.figure(figsize=(20,14))
    
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        
        img = dataset.read_image(idx)        
        _, label = dataset[idx]
        
        plt.imshow(img)
        plt.title(dataset.decode_multi_class(label), fontsize = 16)
        plt.axis('off')
    plt.show()

