import requests
from typing import Literal


class Joke:
    def __init__(self, type: Literal["programming", "misc", "pun", "spooky", "christmas"]):
        self._type = type
    
    def get_joke(self) -> str:
        """Get a joke from the API based on the type."""
        url = f"https://v2.jokeapi.dev/joke/{self._type}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Handle different joke formats (single or two-part jokes)
        if data["type"] == "single":
            return data["joke"]
        elif data["type"] == "twopart":
            return f"{data['setup']}\n{data['delivery']}"
        else:
            return "Could not parse joke"
    
    def __str__(self) -> str:
        """Return the joke as a string."""
        return self.fetch()
        # return self.get_joke()