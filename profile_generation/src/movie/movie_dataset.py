import logging
import os
import pandas as pd

from src.movie.movie_user import MovieUser


class MovieDataset:
    """
    A class used to represent a Movie Dataset.

    Attributes:
        folder (str): The folder containing the dataset files.
        __user_score (pd.DataFrame | None): The user scores dataframe.
        __movie_dict (dict | None): The dictionary of movies.
        __users (pd.DataFrame | None): The dataframe of users.
        __movies (pd.DataFrame | None): The dataframe of movies.
        __ratings (pd.DataFrame | None): The dataframe of ratings.
        __data (dict | None): The dictionary of user data.
        __white_list (list | None): The whitelist of movie IDs.

    Methods:
        user_score: Returns the user scores dataframe.
        movie_dict: Returns the dictionary of movies.
        users: Returns the dataframe of users.
        movies: Returns the dataframe of movies.
        ratings: Returns the dataframe of ratings.
        data: Returns the dictionary of user data.
        __getitem__(user_id): Returns the user data for the given user ID.
        __len__(): Returns the number of users.
        __iter__(): Returns an iterator over the users.
    """

    def __init__(self, folder: str, whitelist: list | None = None):
        """
        Initializes the MovieDataset with the provided folder and whitelist.

        Args:
            folder (str): The folder containing the dataset files.
            whitelist (list, optional): The whitelist of movie IDs. Defaults to None.
        """
        self.folder = folder
        self.__user_score = None
        self.__movie_dict = None
        self.__users = None
        self.__movies = None
        self.__ratings = None
        self.__data = None
        self.__white_list = None

    @property
    def user_score(self):
        """
        Returns the user scores dataframe.

        Returns:
            pd.DataFrame: The user scores dataframe.
        """
        if self.__user_score is None:
            self.__user_score = self.ratings.groupby('user_id')[['movie_id', 'rating']].apply(
                lambda x: list(zip(x['movie_id'], x['rating']))).reset_index().rename(columns={0: 'ratings'})
        return self.__user_score

    @property
    def movie_dict(self):
        """
        Returns the dictionary of movies.

        Returns:
            dict: The dictionary of movies.
        """
        if self.__movie_dict is None:
            self.__movie_dict = self.movies.set_index('movie_id').to_dict(orient='index')
        return self.__movie_dict

    @property
    def users(self):
        """
        Returns the dataframe of users.

        Returns:
            pd.DataFrame: The dataframe of users.
        """
        if self.__users is None:
            data_path = os.path.join(self.folder, "users.dat")
            logging.info(f"Loading users from {data_path}")
            self.__users = pd.read_csv(data_path, sep="::",
                                       names=["user_id", "gender", "age", "occupation", "zip_code"],
                                       encoding='latin-1', engine='python')
            logging.info(f"User shape: {self.__users.shape}")
        return self.__users

    @property
    def movies(self):
        """
        Returns the dataframe of movies.

        Returns:
            pd.DataFrame: The dataframe of movies.
        """
        if self.__movies is None:
            data_path = os.path.join(self.folder, "movies.dat")
            logging.info(F"Loading movies from {data_path}")
            self.__movies = pd.read_csv(data_path, sep="::", names=["movie_id", "title", "genres"],
                                        encoding='latin-1', engine='python')
            logging.info(f"Movie shape: {self.__movies.shape}")
        return self.__movies

    @property
    def ratings(self):
        """
        Returns the dataframe of ratings.

        Returns:
            pd.DataFrame: The dataframe of ratings.
        """
        if self.__ratings is None:
            data_path = os.path.join(self.folder, "ratings.dat")
            logging.info(F"Loading interactions from {data_path}")
            self.__ratings = pd.read_csv(data_path, sep="::",
                                         names=["user_id", "movie_id", "rating", "timestamp"],
                                         encoding='latin-1', engine='python')
            logging.info(f"Interaction shape: {self.__ratings.shape}")
        return self.__ratings

    @property
    def data(self):
        """
        Returns the dictionary of user data.

        Returns:
            dict: The dictionary of user data.
        """
        if self.__data is None:
            self.__data = {}
            for user_id in self.user_score["user_id"]:
                user_desc = self.user_score.loc[self.user_score["user_id"] == user_id]['ratings'].iloc[0]
                if self.__white_list:
                    user_desc = {self.movie_dict[movie_id]['title']: rating for movie_id, rating in user_desc if
                                 movie_id in self.__white_list}
                else:
                    user_desc = {self.movie_dict[movie_id]['title']: rating for movie_id, rating in user_desc}
                user_info = self.users.loc[self.users.user_id == user_id][['age', 'gender']].to_dict(orient='records')[
                    0]
                user = MovieUser(**user_info, rankings=user_desc, id=user_id)
                self.__data[user_id] = user
        return self.__data

    def __getitem__(self, user_id):
        """
        Returns the user data for the given user ID.

        Args:
            user_id (int | slice | list): The user ID or a slice or list of user IDs.

        Returns:
            MovieUser | list[MovieUser]: The user data for the given user ID or a list of user data.
        """
        if isinstance(user_id, int):
            return self.data[user_id]
        elif isinstance(user_id, slice):
            start = user_id.start
            stop = user_id.stop
            step = user_id.step
            if start is None:
                start = 1
            if stop is None:
                stop = len(self)
            if step is None:
                step = 1
            return [self[idx] for idx in range(start, stop, step)]
        elif isinstance(user_id, list):
            return [self[idx] for idx in user_id]
        else:
            raise ValueError("Invalid type for user_id")

    def __len__(self):
        """
        Returns the number of users.

        Returns:
            int: The number of users.
        """
        return len(self.users)

    def __iter__(self):
        """
        Returns an iterator over the users.

        Yields:
            MovieUser: The next user in the dataset.
        """
        for user_id in self.users['user_id']:
            yield self[user_id]
