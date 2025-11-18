import os
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize DeepSeek model
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

@tool
def analyze_stock(ticker: str) -> dict:
    """
    Analyze a stock across multiple timeframes and provide comprehensive data for buy/sell decision.
    Returns today's data, weekly trend, yearly trend, and technical indicators.
    
    Args:
        ticker: Stock ticker symbol
    """
    try:
        days_offset = 7
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Calculate reference date (offset from today)
        reference_date = datetime.now() - timedelta(days=days_offset)
        end_date = reference_date
        
        # Get different time periods relative to the offset date
        start_today = end_date - timedelta(days=1)
        start_week = end_date - timedelta(days=5)
        start_month = end_date - timedelta(days=30)
        start_year = end_date - timedelta(days=365)
        
        today = stock.history(start=start_today.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        week = stock.history(start=start_week.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        month = stock.history(start=start_month.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        year = stock.history(start=start_year.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if today.empty or week.empty or year.empty:
            return {"status": "error", "message": f"No data found for {ticker}"}
        
        # Current data
        current_price = today.iloc[-1]['Close']
        
        # Calculate trends
        day_change = ((current_price - today.iloc[0]['Open']) / today.iloc[0]['Open']) * 100
        week_change = ((current_price - week.iloc[0]['Close']) / week.iloc[0]['Close']) * 100
        month_change = ((current_price - month.iloc[0]['Close']) / month.iloc[0]['Close']) * 100
        year_change = ((current_price - year.iloc[0]['Close']) / year.iloc[0]['Close']) * 100
        
        # Calculate moving averages
        week_avg = week['Close'].mean()
        month_avg = month['Close'].mean()
        
        # Simple momentum indicators
        above_week_avg = current_price > week_avg
        above_month_avg = current_price > month_avg
        
        # Volume analysis
        avg_volume = month['Volume'].mean()
        current_volume = today.iloc[-1]['Volume']
        volume_ratio = current_volume / avg_volume
        
        # 52-week high/low context
        year_high = year['High'].max()
        year_low = year['Low'].min()
        price_position = ((current_price - year_low) / (year_high - year_low)) * 100
        
        return {
            "status": "success",
            "ticker": ticker.upper(),
            "company_name": info.get('longName', 'N/A'),
            "current_price": round(current_price, 2),
            
            # Price changes
            "day_change_percent": round(day_change, 2),
            "week_change_percent": round(week_change, 2),
            "month_change_percent": round(month_change, 2),
            "year_change_percent": round(year_change, 2),
            
            # Moving averages
            "week_average": round(week_avg, 2),
            "month_average": round(month_avg, 2),
            "above_week_avg": above_week_avg,
            "above_month_avg": above_month_avg,
            
            # Volume
            "volume_ratio": round(volume_ratio, 2),  # >1 means higher than average
            "current_volume": int(current_volume),
            
            # Year context
            "52_week_high": round(year_high, 2),
            "52_week_low": round(year_low, 2),
            "price_position_percent": round(price_position, 2),  # Where in 52w range (0-100%)
            
            # Additional info
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error analyzing stock: {str(e)}"}


def create_stock_agent():
    """
    Create and return a stock trading advisor agent.
    Can be imported and used in other programs.
    """
    return create_agent(
        model=llm,
        tools=[analyze_stock],
        system_prompt="""You are a stock trading advisor. When a user asks about a stock, use the analyze_stock tool and provide a clear BUY or SELL recommendation.

Base your recommendation on:
1. **Trend Analysis**: 
   - Positive day/week/month/year changes = Bullish signal
   - Negative changes = Bearish signal
   
2. **Moving Averages**:
   - Price above week and month averages = Bullish
   - Price below averages = Bearish
   
3. **Momentum**:
   - Volume ratio > 1.2 with positive price action = Strong buying interest
   - Volume ratio > 1.2 with negative price action = Strong selling pressure
   
4. **Position in Range**:
   - Price position > 70% (near 52-week high) = May be overbought
   - Price position < 30% (near 52-week low) = May be oversold/opportunity

   The customer wants to hold the stock for one week so decie wehter you think the stock will go up or down in that timeframe.
Always structure your response as:

<Recommendation: BUY** or **Recommendation: SELL>

and only this sentence
""",
    )


# Create the agent instance for import
agent = create_stock_agent()


# Run the agent
if __name__ == "__main__":

    ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip()

    print(f"\n{'='*60}")
    print(f"Analyzing {ticker.upper()}...")
    print('='*60)
    
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Should I buy or sell {ticker}?"
            }
        ]
    })
    
    print(result["messages"][-1].content)