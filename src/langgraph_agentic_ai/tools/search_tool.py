from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

def get_tools():
    """
    Return the list of tools to be used in Chatbot
    """
    tavily_search = TavilySearch(max_results=2)
    
    @tool
    def search_web(query: str) -> str:
        """
        Search the web for information using Tavily Search.
        
        Args:
            query: The search query string
            
        Returns:
            Search results as a string
        """
        try:
            results = tavily_search.invoke({"query": query})
            return str(results)
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    tools = [search_web]
    return tools

def create_tool_node(tools):
    """
    Creates and returns a Tool Node for the graph
    """
    return ToolNode(tools)