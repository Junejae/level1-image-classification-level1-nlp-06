from dataset import MaskLabels, GenderLabels, AgeLabels

from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image # tensor to pil_image
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from PIL import Image
from sklearn.decomposition import PCA


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
    

class PCA():
    def __init__(self, dataset, train_indices, n_components):
        imgs = []
        for i in train_ss.indices:
            img = np.array(Image.open(ds.image_paths[i]).convert('L'))
            imgs.append(np.array(img))
        imgs = np.array(imgs)
        self.n_samples, self.h, self.w = imgs.shape

        imgs = np.reshape(imgs, (n_samples, h*w)) # 차원 변환

        self.pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(imgs)
        print(f"pca is fitted in {time() - t0:.0f}s")
        print(f'Explained variation per principal component: \n{pca.explained_variance_ratio_}')
        self.eigenfaces = pca.components_.reshape((n_components, h, w)) # 차원 변환
    
    def show_pred_by_pca(self, dataset, train_idx):
        plt.subplot(1, 2, 1)
        img, _ = dataset[i]
        plt.imshow(img.reshape((self.h, self.w)))
        plt.subplot(1, 2, 2)
        plt.imshow(self.eigenfaces[train_idx].reshape((self.h, self.w)), cmap=plt.cm.gray)
        plt.show()
        
    def transform(self, images):
        return self.pca.transform(images)
    
    def extend_dim(self, pca_images, labels):
        extend_ds = []
        for x, y in zip(pca_images, labels):
            extend_ds.append((torch.tensor([[x]]*3, dtype=float).float().cuda(), y))
        return extend_ds


