from collections import defaultdict
from pydantic import BaseModel

from.music_item import MusicItem

class MusicUser(BaseModel):
    id: str
    ratings: dict[MusicItem, int] | None = None
    new_prompt: bool = False
    timestamps: dict[MusicItem, int] | None = None
    description: str | None = None
    embedding: list[float] | None = None

    def add(self, item: MusicItem, rating: int, timestamp: int = None):
        if self.ratings is None:
            self.ratings = {}
            self.timestamps = {}
        if item not in self.ratings or timestamp > self.timestamps[item]:
            self.ratings[item] = rating
            self.timestamps[item] = timestamp


    def dict(self, *args, **kwargs):
        new_dict = {"id": self.id, "ratings": {str(k): v for k, v in self.ratings.items()},
                    "prompt": self.prompt(),
                    "description": self.description,
                    "embedding": self.embedding}
        return new_dict


    def prompt(self):
        if self.new_prompt:
            return self.prompt_v2()
        return self.prompt_v1()

    def prompt_v1(self, ):
        desc = f"I have rated {len(self.ratings)} items. "
        desc += ", ".join([f"{item.title} of {item.brand} in {item.categories if item.categories else 'unknown'} "
                           f"category: {rating}" for item, rating in self.ratings.items()])
        return desc

    def prompt_v2(self):
        marks = defaultdict(list)
        for item, rating in self.ratings.items():
            marks[rating].append(f"{item.title} of {item.brand} "
                                 f"in {item.categories if item.categories else 'unknown'} category")

        desc = (f"I found following items excellent and rate theme five of five: {', '.join(marks[5])}. " if marks[5] else "",
                f"Items that I found good and rate them four of five: {', '.join(marks[4])}. " if marks[4] else "",
                f"Items that I found average and rate them three of five: {', '.join(marks[3])}. " if marks[3] else "",
                f"Items that I found below average and rate them two of five: {', '.join(marks[2])}. " if marks[2] else "",
                f"Items that I found terrible and rate them one of five: {', '.join(marks[1])}. " if marks[1] else "")
        desc = "\n".join([x for x in desc if x ])
        return desc

    def __hash__(self):
        return hash(self.id)