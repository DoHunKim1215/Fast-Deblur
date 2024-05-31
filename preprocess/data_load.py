import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path)

    loader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur', ''))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blur = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        sharp = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx]))

        blur = F.to_tensor(blur)
        sharp = F.to_tensor(sharp)

        if self.is_test:
            name = self.image_list[idx]
            return blur, sharp, name

        return blur, sharp

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPEG']:
                raise ValueError
