from pydantic import ValidationError
from tqdm import tqdm
import logging
import os
import pandas as pd

from src.music.music_item import MusicItem
from src.music.music_user import MusicUser


class MusicDataset:
    def __init__(self, folder: str):
        self.folder = folder
        self.__items = None
        self.__interactions = None
        self.__users = None


    @property
    def items(self):
        if self.__items is None:
            self.__items = self.get_item_dict(self.folder)
        return self.__items

    @property
    def users(self):
        if self.__users is None:
            self.__users = self.get_user_dict(self.folder, self.items)
        return self.__users

    @classmethod
    def get_user_dict(cls, folder: str, item_dict: dict[str, MusicItem]) -> dict[str, MusicUser]:
        data_path = os.path.join(folder, "Amazon_CDs_and_Vinyl.inter")
        logging.info(f"Loading interactions from {data_path}")
        interactions = pd.read_csv(data_path, sep='\t')

        logging.info(f"Interaction shape: {interactions.shape}")
        interactions = interactions.rename(columns={x: x.split(":")[0] for x in interactions.columns})
        inter_dict = interactions.to_dict(orient='index')

        logging.info("Processing interactions")
        users = {}
        for _, v in tqdm(inter_dict.items()):
            if v['user_id'] not in users:
                users[v['user_id']] = MusicUser(id=v['user_id'])
            try:
                idx = MusicItem.validate_ids(v['item_id'])
                users[v['user_id']].add(item_dict[idx], v['rating'], v['timestamp'] if 'timestamp' in v else 0)
            except KeyError:
                continue

        return users

    @classmethod
    def get_item_dict(cls, folder: str) -> dict[str,MusicItem]:
        data_path = os.path.join(folder, "Amazon_CDs_and_Vinyl.item")
        logging.info(f"Loading items from {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        df = pd.read_csv(data_path, sep='\t')
        df = df.rename(columns={x: x.split(":")[0] for x in df.columns})
        df = df.set_index('item_id').fillna("").to_dict(orient='index')
        items = {}
        for k, v in df.items():
            try:
                item = MusicItem(id=k,
                                 title=v["title"],
                                 categories=v["categories"],
                                 brand=v["brand"],
                                 sales_type=v["sales_type"])
                items[item.id] = item
            except ValidationError as e:
                print(k)
        return items

    def __getitem__(self, item):
        return self.users[item]

    def __iter__(self):
        return iter(self.users.values())

    def __len__(self):
        return len(self.users)