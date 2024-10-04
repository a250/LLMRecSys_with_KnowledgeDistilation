from openai import OpenAI, RateLimitError
import backoff


class OpenAIAdapter:
    """
    A class used to interact with the OpenAI API.

    Attributes:
        client (OpenAI): The OpenAI client initialized with the provided API token.
        model (str): The model to use for generating completions. Defaults to "gpt-3.5-turbo".

    Methods:
        send_prompt(messages: list): Sends a list of messages to the OpenAI API and returns the response.
        get_embedding(text: str, model: str = "text-embedding-3-small"): Gets the embedding for the provided text.
    """

    def __init__(self, token: str, model: str | None = None):
        """
        Initializes the OpenAIAdapter with the provided API token and model.

        Args:
            token (str): The API token for authenticating with the OpenAI API.
            model (str, optional): The model to use for generating completions. Defaults to "gpt-3.5-turbo".
        """
        self.client = OpenAI(api_key=token)
        self.model = model if model else "gpt-3.5-turbo"

    @backoff.on_exception(backoff.expo, RateLimitError, max_time=600)
    def send_prompt(self, messages: list):
        """
        Sends a list of messages to the OpenAI API and returns the response.

        Args:
            messages (list): A list of messages to send to the OpenAI API.

        Returns:
            dict: The response from the OpenAI API.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response

    def get_embedding(self, text: str, model="text-embedding-3-small"):
        """
        Gets the embedding for the provided text.

        Args:
            text (str): The text to get the embedding for.
            model (str, optional): The model to use for generating the embedding. Defaults to "text-embedding-3-small".

        Returns:
            list: The embedding of the provided text.
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding