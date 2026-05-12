import os
import random

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class TripletFaceDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir
    ):

        self.df = pd.read_csv(csv_file)

        self.image_dir = image_dir


        self.transform = transforms.Compose([

            transforms.Resize((64, 64)),

            transforms.ToTensor()
        ])


        # PRECOMPUTE GROUPS
        self.grouped = {}

        for idx, row in self.df.iterrows():

            key = (
                row["gender"],
                row["race"]
            )

            if key not in self.grouped:

                self.grouped[key] = []

            self.grouped[key].append(idx)


        self.all_indices = list(
            range(len(self.df))
        )


    def __len__(self):

        return len(self.df)


    def load_image(self, image_name):

        path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(path).convert(
            "RGB"
        )

        image = self.transform(image)

        return image


    def __getitem__(self, idx):

        anchor_row = self.df.iloc[idx]


        anchor_key = (

            anchor_row["gender"],

            anchor_row["race"]
        )


        # POSITIVE SAMPLE
        positive_idx = random.choice(
            self.grouped[anchor_key]
        )

        while positive_idx == idx:

            positive_idx = random.choice(
                self.grouped[anchor_key]
            )


        # NEGATIVE SAMPLE
        negative_idx = random.choice(
            self.all_indices
        )

        negative_row = self.df.iloc[
            negative_idx
        ]


        while (

            negative_row["gender"]
            == anchor_row["gender"]

            and

            negative_row["race"]
            == anchor_row["race"]

        ):

            negative_idx = random.choice(
                self.all_indices
            )

            negative_row = self.df.iloc[
                negative_idx
            ]


        positive_row = self.df.iloc[
            positive_idx
        ]


        anchor_img = self.load_image(
            anchor_row["image"]
        )

        positive_img = self.load_image(
            positive_row["image"]
        )

        negative_img = self.load_image(
            negative_row["image"]
        )


        return (

            anchor_img,

            positive_img,

            negative_img
        )