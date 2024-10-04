from typing import Literal

from pydantic import BaseModel


class MovieUser(BaseModel):
    """A class used to represent a Movie User.

    Attributes:
        id (int): The unique identifier for the user.
        gender (Literal["M", "F"]): The gender of the user.
        age (int): The age of the user.
        rankings (dict[str, int] | None): The movie rankings given by the user.
        description (str | None): The description of the user.
        embedding (list[float] | None): The embedding of the user.
        AGE_DICT (dict[int, str]): A dictionary mapping age ranges to descriptions.

    Methods:
        prompt(): Generates a description prompt for the user.
        __hash__(): Returns the hash of the user id.
    """

    id: int
    gender: Literal["M", "F"]
    age: int
    rankings: dict[str, int] | None = None
    description: str | list[dict[str, str]] | None = None
    embedding: list[float] | None = None
    AGE_DICT: dict[int, str] = {1: "Under 18",
                                18: "18-24",
                                25: "25-34",
                                35: "35-44",
                                45: "45-49",
                                50: "50-55",
                                56: "56+"}

    def prompt(self):
        """Generates a description prompt for the user.

        Returns:
            str: The description prompt for the user.
        """
        gender = "Male" if self.gender == "M" else "Female"
        desc = f"I am a {gender} of age {self.AGE_DICT[self.age]} and I rank movies as follows: "
        movie_desc = ", ".join([f"{movie}: {rating}" for movie, rating in self.rankings.items()])
        desc += movie_desc
        return desc

    def __hash__(self):
        """Returns the hash of the user id.

        Returns:
            int: The hash of the user id.
        """
        return hash(self.id)

    def dict(self, *args, **kwargs) -> dict:
        data = self.model_dump(exclude={"AGE_DICT"})
        data["prompt"] = self.prompt()
        return data
