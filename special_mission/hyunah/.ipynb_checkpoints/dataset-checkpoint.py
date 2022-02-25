from torch.utils.data import Dataset, DataLoader
from PIL import Image


def convert_to_dataloader(dataset, BATCH_SIZE, IsShuffle):
    return DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle=IsShuffle
    )


class TrainDataset(Dataset):
    # input: image_list, target_list
    def __init__(self, X, y, transform=None):
        self.image_paths = X
        self.target = y
        self.transform = transform
    
    # output: PIL_image, label
    def __getitem__(self, index):
        images = []
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        
        return (image, self.target[index])
    
    def __len__(self):
        return len(self.image_paths)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)