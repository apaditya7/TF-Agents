from crewai.tools import tool
import requests
import os
from typing import Dict, Optional

@tool("Google Fact Check")
def google_fact_check(query: str) -> str:
    """Verify claims using Google's Fact Check API to find fact checks from reputable sources."""
    api_key = "AIzaSyC4hRxckC42eHqRW_Zci60-OzL4JE60AwA" or os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        return "Error: Google Fact Check API key is required"
    
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": api_key,
        "query": query,
        "languageCode": "en-US"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return format_fact_check_results(data)
            
    except requests.RequestException as e:
        return f"Error performing fact check: {str(e)}"

def format_fact_check_results(data: Dict) -> str:
    """Format the fact check API response into a readable format."""
    if "claims" not in data or not data["claims"]:
        return "No fact checks found for this claim."
    
    results = []
    for claim in data.get("claims", []):
        claim_text = claim.get("text", "Unknown claim")
        
        for review in claim.get("claimReview", []):
            publisher = review.get("publisher", {}).get("name", "Unknown source")
            rating = review.get("textualRating", "No rating provided")
            review_url = review.get("url", "")
            title = review.get("title", "")
            
            result = f"Claim: {claim_text}\n"
            result += f"Publisher: {publisher}\n"
            result += f"Rating: {rating}\n"
            
            if title:
                result += f"Title: {title}\n"
            
            if review_url:
                result += f"Source: {review_url}\n"
            
            results.append(result)
    
    return "\n\n".join(results)



@tool("Web Search")
def serper_search(query: str) -> str:
    """Search the web for current information on a topic or claim using Serper API."""
    api_key = "bd0cebf64a45d35de36a6e0c1d6faeb7ec1bdec0" or os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: Serper API key is required"
    
    base_url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "gl": "us",
        "hl": "en"
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Format the response
        return format_search_results(data)
            
    except requests.RequestException as e:
        return f"Error performing search: {str(e)}"

def format_search_results(data: Dict) -> str:
    """Format the search API response into a readable format."""
    if "organic" not in data or not data["organic"]:
        return "No search results found."
    
    results = []
    for idx, result in enumerate(data.get("organic", [])[:5], 1):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No description available")
        link = result.get("link", "")
        
        formatted = f"{idx}. {title}\n"
        formatted += f"   {snippet}\n"
        
        if link:
            formatted += f"   Source: {link}\n"
        
        results.append(formatted)
    
    return "\n\n".join(results)


@tool("Comprehensive Research")
def combined_research(query: str) -> str:
    """Research a topic using both web search and fact checking to provide verified information."""
    search_results = serper_search(query)
    fact_check_results = google_fact_check(query)
    
    combined = "## WEB SEARCH RESULTS\n\n"
    combined += search_results
    combined += "\n\n## FACT CHECK RESULTS\n\n"
    combined += fact_check_results
    
    return combined