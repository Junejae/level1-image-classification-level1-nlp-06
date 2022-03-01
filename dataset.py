import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter, RandomPerspective, RandomHorizontalFlip, RandomRotation
# customize
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class JuneCustomDataset(Dataset):
    def __init__(self, data_dir, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2, ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.num_classes = 18

        self.data_info = self.preprocessedData(self.data_dir) # preprocessFunction()
        self.img_paths = self.data_info.img_path.tolist()
        self.transform = None
        self.transform_june = JuneCustomAug(self.mean, self.std)
        self.label_arr = self.data_info.label.tolist()

        self.data_len = len(self.data_info.img_path)
        
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.label_arr[index]

        if self.transform:
            """ if label % 3 == 2:
                image = self.transform_june(image)
            else:
                image = self.transform(image) """

            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.data_len

    # split in my way
    def split_dataset(self) -> Tuple[Subset, Subset]:
        """ sss = ShuffleSplit(n_splits=1, test_size=self.val_ratio)
        
        train_index, val_index = [], []

        for t, v in sss.split(range(self.data_len)):
            train_index = t
            val_index = v
       
        return Subset(self, train_index), Subset(self, val_index) """

        full = list(range(self.data_len))

        div_point = int(self.data_len * (1-self.val_ratio))

        train_index, val_index = full[:div_point], full[div_point:]

        return Subset(self, train_index), Subset(self, val_index)

    # re-labeling process
    def preprocessedData(self, train_dir):
        prepro_data_info = pd.DataFrame(columns={'id','img_path','race','mask','gender','age','label'})

        all_id, all_path, all_race, all_mask, all_age, all_gender, all_label = [],[],[],[],[],[],[]

        for absolute_path in glob(train_dir + "/*/*"):

            split_list = absolute_path.split("/")
            img_name = split_list[-1]
            img_path = split_list[-2]

            path_split = img_path.split("_")

            img_id = path_split[0]
            img_gender = 0 if path_split[1] == "male" else 1
            img_race = path_split[2]
            img_age = min(2, int(path_split[3]) // 30)

            img_mask = 0
            if 'incorrect' in img_name:
                img_mask = 1
            elif 'normal' in img_name:
                img_mask = 2

            # -- 결측치 교정 시작
            # -- Swap Gender
            if img_id in ['000225','000664','000767','001498-1','001509','003113','003223','004281','004432','005223','006359',
                    '006360','006361','006362','006363','006364','006424','000667','000725','000736','000817','003780','006504']:
                temp = 0 if img_gender == 1 else 1
                img_gender = temp
            
            # -- Change Age to ~29
            if img_id in ['001009','001064','001637','001666','001852']:
                img_age = 0
            
            # -- Change Age to 60~
            if img_id in ['004348']: # 고민거리, 이분은 액면가는 폭삭 늙으셨지만 59세로 찍혀 있는데 과연 이걸 60대 노인 취급해도 될지 안 될지...
                img_age = 2

            # -- Correct Mask Status, normal <-> incorrect
            if img_id in ['000020','004418','005227']:
                if img_mask != 0:
                    temp = 1 if img_mask == 2 else 2
                    img_mask = temp
            # -- 결측치 교정 끝

            # oversampling
            n = 1
            if (img_age == 1 and img_gender == 0):
                n *= 2
            if img_age == 2:
                n *= 8
            if img_mask != 0:
                n *= 4
            
            for _ in range(n):
                all_id.append(img_id)
                all_path.append(absolute_path)
                all_race.append(img_race)
                all_mask.append(img_mask)
                all_gender.append(img_gender)
                all_age.append(img_age)
                all_label.append(img_mask*6 + img_gender*3 + img_age)
            

        prepro_data_info['id'] = all_id
        prepro_data_info['img_path'] = all_path
        prepro_data_info['race'] = all_race
        prepro_data_info['mask'] = all_mask
        prepro_data_info['gender'] = all_gender
        prepro_data_info['age'] = all_age
        prepro_data_info['label'] = all_label
        
        return prepro_data_info

class JuneCustomDataset2(Dataset):
    def __init__(self, data_info, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.num_classes = 18

        self.data_info = data_info
        self.img_paths = self.data_info.img_path.tolist()
        self.transform = None
        self.label_arr = self.data_info.label.tolist()

        self.data_len = len(self.data_info.img_path)
        
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.label_arr[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.data_len

class JuneCustomAug:
    def __init__(self, mean, std, resize=[512, 384], **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            RandomPerspective(distortion_scale=0.2, p=0.8), 
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class JuneCustomAug2:
    def __init__(self, mean, std, resize=[512, 384], **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class JuneCustomAug3:
    def __init__(self, mean, std, resize=[512, 384], **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            RandomRotation(degrees=20),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class JuneCustomAugSOTA:
    def __init__(self, mean, std, resize=[512, 384], **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5,0.5,0.5), std=(0.2,0.2,0.2)), # not using CIFAR10 standard
        ])

    def __call__(self, image):
        return self.transform(image)