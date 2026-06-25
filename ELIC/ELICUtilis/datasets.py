# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# 支持的图片格式
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories.

    Supports both flat and nested directory structures, e.g.:

    .. code-block::

        Flat (original):
        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png

        Nested (e.g. D-Fire detection dataset):
        - rootdir/
            - train/
                - images/
                    - img000.jpg
                - labels/
                    - img000.txt
            - test/
                - images/
                    - img000.jpg

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{splitdir}"')

        # 递归搜索所有子目录下的图片文件，自动跳过标注文件等非图片内容
        self.samples = [
            f for f in sorted(splitdir.rglob("*"))
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f'No valid images found in "{splitdir}". '
                f'Supported formats: {", ".join(sorted(VALID_EXTENSIONS))}'
            )

        print(f"[ImageFolder] Found {len(self.samples)} images in '{splitdir}'")

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)