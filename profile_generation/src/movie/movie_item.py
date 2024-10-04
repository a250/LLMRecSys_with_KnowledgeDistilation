from pydantic import BaseModel


class MovieItem(BaseModel):
    """A class used to represent a Movie Item.

    Attributes:
        title (str): The title of the movie.

    Methods:
        __str__(): Returns the string representation of the movie title.
        __hash__(): Returns the hash of the movie title.
        __repr__(): Returns the string representation of the movie title.
        __eq__(other): Checks if the title of this movie is equal to the title of another movie.
        __ne__(other): Checks if the title of this movie is not equal to the title of another movie.
        __lt__(other): Checks if the title of this movie is less than the title of another movie.
        __le__(other): Checks if the title of this movie is less than or equal to the title of another movie.
        __gt__(other): Checks if the title of this movie is greater than the title of another movie.
        __ge__(other): Checks if the title of this movie is greater than or equal to the title of another movie.
    """

    title: str

    def __str__(self):
        """
        Returns the string representation of the movie title.

        Returns:
            str: The title of the movie.
        """
        return f"{self.title}"

    def __hash__(self):
        """
        Returns the hash of the movie title.

        Returns:
            int: The hash of the movie title.
        """
        return hash(str(self))

    def __repr__(self):
        """
        Returns the string representation of the movie title.

        Returns:
            str: The title of the movie.
        """
        return f"{self.title}"

    def __eq__(self, other):
        """
        Checks if the title of this movie is equal to the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if the titles are equal, False otherwise.
        """
        return self.title == other.title

    def __ne__(self, other):
        """
        Checks if the title of this movie is not equal to the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if the titles are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Checks if the title of this movie is less than the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if this title is less than the other title, False otherwise.
        """
        return self.title < other.title

    def __le__(self, other):
        """
        Checks if the title of this movie is less than or equal to the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if this title is less than or equal to the other title, False otherwise.
        """
        return self.title <= other.title

    def __gt__(self, other):
        """
        Checks if the title of this movie is greater than the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if this title is greater than the other title, False otherwise.
        """
        return self.title > other.title

    def __ge__(self, other):
        """
        Checks if the title of this movie is greater than or equal to the title of another movie.

        Args:
            other (MovieItem): The other movie to compare with.

        Returns:
            bool: True if this title is greater than or equal to the other title, False otherwise.
        """
        return self.title >= other.title
