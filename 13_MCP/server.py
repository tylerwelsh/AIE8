from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller
from joke import Joke

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def get_joke(joke_type: str = "programming") -> str:
    """Get a joke of the given type. Available types: programming, misc, pun, spooky, christmas"""
    print(f"Getting joke of type: {joke_type}") 
    joke = Joke(joke_type)
    joke = joke.get_joke()
    return str(joke)

if __name__ == "__main__":  
    print("Starting MCP server...")
    mcp.run(transport="stdio")