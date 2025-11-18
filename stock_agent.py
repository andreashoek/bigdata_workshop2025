import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool  # FIX: Import from langchain_core
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize DeepSeek model using OpenAI-compatible interface
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# Define the stock lookup tool
@tool
def get_stock_data(ticker: str) -> dict:
    """
    Fetch stock data for a given ticker symbol.
    Use this when the user asks about a specific company's stock information.
    """
    massive_api_key = os.getenv("MASSIVE_API_KEY")  # Your Massive.com API key
    url = "https://api.massive.com/v3/reference/tickers"
    
    params = {
        "market": "stocks",
        "active": "true",
        "order": "asc",
        "limit": 100,
        "sort": "ticker",
        "apiKey": massive_api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Filter results for the requested ticker
        if "results" in data:
            matching_stocks = [
                stock for stock in data["results"] 
                if stock.get("ticker", "").upper() == ticker.upper()
            ]
            
            if matching_stocks:
                return {
                    "status": "found",
                    "ticker": ticker.upper(),
                    "data": matching_stocks[0]
                }
            else:
                return {
                    "status": "not_found",
                    "ticker": ticker.upper(),
                    "message": f"Ticker {ticker} not found in active stocks"
                }
        
        return {"status": "error", "message": "Invalid API response"}
        
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API request failed: {str(e)}"}

# Create the agent
agent = create_agent(
    model=llm,
    tools=[get_stock_data],
    system_prompt="You are a helpful stock information assistant. When users ask about a company's stock, use the get_stock_data tool to fetch information and explain the results clearly."
)

# Run the agent
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Can you get me information about Apple stock (AAPL)?"
            }
        ]
    })
    
    print(result["messages"][-1].content)