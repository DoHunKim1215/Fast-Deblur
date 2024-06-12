import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path)

    transform = transforms.Compose([
        transforms.CenterCrop((960, 960)),
        transforms.ToTensor()
    ])

    loader = DataLoader(
        DeblurDataset(image_dir, is_test=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, ''))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        blur = Image.open(os.path.join(self.image_dir, self.image_list[idx]))

        if self.transform is not None:
            blur = self.transform(blur)

        if self.is_test:
            name = self.image_list[idx]
            return blur, name

        return blur

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPEG']:
                raise ValueError
