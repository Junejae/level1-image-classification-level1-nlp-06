import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image # tensor to pil_image

def show_images(dataset, idxs, rows=1, cols=7):
    plt.figure(figsize=(20,14))

    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        img, label = dataset[idx]
        plt.imshow(to_pil_image(img))
        plt.title(label, fontsize = 16)
        plt.axis('off')
    plt.show()