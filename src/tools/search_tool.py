"""Search tool using Tavily API"""
from tavily import TavilyClient
from typing import List, Dict
from src.utils.config import Config
from src.utils.logger import logger


class SearchTool:
    """Wrapper for Tavily search functionality"""
    
    def __init__(self):
        self.client = TavilyClient(api_key=Config.TAVILY_API_KEY)
    
    def search(self, query: str, topic: str = "general") -> List[Dict]:
        """
        Perform a search query using Tavily API
        
        Args:
            query: Search query string
            topic: Topic context ("general", "news", "research")
        
        Returns:
            List of search results with content, URLs, and relevance
        """
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                topic=topic,
                max_results=10
            )
            logger.info(f"Search query '{query}' returned {len(response.get('results', []))} results")
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            return []
    
    def search_market(self, startup_idea: str) -> List[Dict]:
        """Search for market information about a startup idea"""
        return self.search(f"market for {startup_idea} industry trends", topic="general")
    
    def search_competitors(self, startup_idea: str) -> List[Dict]:
        """Search for competitor information"""
        return self.search(f"companies competitors {startup_idea}", topic="general")
    
    def search_technology(self, tech_requirements: str) -> List[Dict]:
        """Search for technology and technical feasibility information"""
        return self.search(f"how to build {tech_requirements} technical stack", topic="general")
