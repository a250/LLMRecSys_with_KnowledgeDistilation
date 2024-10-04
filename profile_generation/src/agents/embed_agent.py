from typing import Literal
from functools import cache
from textwrap import dedent
import yaml

from src.movie.movie_user import MovieUser
from src.music.music_user import MusicUser
from src.agents.openai_adapter import OpenAIAdapter

class EmbedAgent:
    """
    A class used to interact with the embedding agent.

    Attributes:
        agent (OpenAIAdapter): The adapter for the OpenAI API.

    Methods:
        get_user_description(item, test: bool = False): Gets the user description.
        encode_description(description: str): Encodes the description into embeddings.
        encode_user(user, test: bool = False): Encodes the user into embeddings.
    """

    def __init__(self, agent: Literal["openai"], model: str | None = None) -> None:
        """
        Initializes the EmbedAgent with the provided agent and model.

        Args:
            agent (Literal["openai"]): The agent to use for embeddings.
            model (str, optional): The model to use for generating embeddings. Defaults to None.
        """
        if agent == "openai":
            with open('token.yaml', 'r') as file:
                token = yaml.safe_load(file)['openai']
            self.agent = OpenAIAdapter(token=token, model=model)

    def get_user_description(self, item, test: bool = False) -> str:
        """
        Gets the user description. This method should be implemented by the subclass.

        Args:
            item: The item to get the description for.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            str: The description of the user.
        """
        pass

    @cache
    def encode_description(self, description: str) -> list[float]:
        """
        Encodes the description into embeddings.

        Args:
            description (str): The description to encode.

        Returns:
            list[float]: The embeddings of the description.
        """
        return self.agent.get_embedding(description)

    def encode_user(self, user, test: bool = False) -> list[float]:
        """
        Encodes the user into embeddings.

        Args:
            user: The user to encode.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            list[float]: The embeddings of the user.
        """
        description = self.get_user_description(user, test)
        return self.encode_description(description)


class EmbedAgentMovie(EmbedAgent):
    """A class used to interact with the embedding agent specifically for movie users.

    Methods:
        get_user_description(user: MovieUser, test: bool = False): Gets the user description.
    """

    @cache
    def get_user_description(self, user: MovieUser, test: bool = False) -> str | list:
        """Gets the user description.

        Args:
            user (MovieUser): The user to get the description for.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            str | list: The description of the user or the prompt if in test mode.
        """
        prompt = [
            {
                "role": "system",
                "content": dedent(
                    """
                    You will be presented with user ratings and your job is to provide a general
                    summarization of users preferences. You should pay attention to movie genres, its release year,
                    main cast, director, awards, critical acclaims, and other relevant information.
                    The response should be split into several parts: first should provide the analysis of movie
                    preferences based on its year. The second should be focused on genres and plot twists.
                    The third one should describe preferences in cast and directors.
                    The fourth should characterise users choice based on correlation between users ranking and critical
                    acclamation and reviews of movies. The fifth paragraph should describe the movies with user dislike
                    based on assigned rating 3 and below. What do these low-ranked movies have in common?
                    Provided user's profile as a general description of the user's preferences should avoid mentioning
                    the actual movies and ratings or user personal data. Try to predict favourite users actor, director,
                    genre, etc. The user profile should be useful for further movie recommendations and it should be at
                    least 5 paragraphs long. The description should include characteristics of the most relevant genres
                    to the user. Avoid mentioning specific movies in response.
                    """).replace("\n", " ")
            },
            {
                "role": "user",
                "content": user.prompt()
            }
        ]
        if test:
            return prompt
        return self.agent.send_prompt(prompt).choices[0].message.content


class EmbedAgentMusic(EmbedAgent):

    @cache
    def get_user_description(self, user: MusicUser, test: bool = False):

        prompt = [
            {
                "role": "system",
                "content": dedent("You will be presented with user ratings and your job is to provide a general"
                                  "summarization of users preferences.  You should pay attention to misic genres,"
                                  "its release year, musicians, used instruments, text meaning, awards,"
                                  "critical acclaims, and other relevant information. The response should be splitted"
                                  "on several parts, first should provide the analysis of music albums preferences"
                                  "based on its year. The second should be focused on genres and lyrics."
                                  "The third one should describe preferences in musicians and instruments."
                                  "The fourth should characterise users choice based on correlation between users"
                                  "ranking and critical acclamation and reviews of music a,bums. The fifth paragraph"
                                  "should describe the albums which user dislikes based on assigned rating 3 and below."
                                  "What this low-ranked albums have in common? Provided user's profile as a general"
                                  "description of the user's preferences should avoid mentioning the actual albums and"
                                  "ratings or user personal data. Try to predict favourite users musicians, bands,"
                                  "genre, etc. The user profile should be useful for further music recommendations and"
                                  "it should be at least 5 paragraph long. The description should include"
                                  "characteristic of the most relevant genres to the user."
                                  "A void mentioning specific albums in response.").replace("\n", " ")
            },
            {
                "role": "user",
                "content": user.prompt()}
        ]
        if test:
            return prompt
        return  self.agent.send_prompt(prompt).choices[0].message.content
