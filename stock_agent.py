import os
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from langsmith import traceable


# Load environment variables
load_dotenv()

# Initialize DeepSeek model
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# Initialize FinBERT for sentiment analysis
print("Loading FinBERT model...")
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert.eval()  # Set to evaluation mode
print("FinBERT model loaded successfully!")


@traceable(run_type="llm")
def get_sentiment_score(text):
    """
    Analyze sentiment of financial text using FinBERT.
    Returns a sentiment score between -1 (negative) and 1 (positive).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    try:
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = finbert(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to numpy
        probs = predictions[0].numpy()
        
        # Labels: 0=neutral, 1=positive, 2=negative
        # Calculate sentiment score: positive - negative
        sentiment_score = float(probs[1] - probs[2])
        
        return sentiment_score
    
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0.0

@tool
@traceable
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


@tool
@traceable
def analyze_stock_sentiment(ticker: str) -> dict:
    """
    Analyze sentiment of recent news for a stock using FinBERT.
    Returns sentiment score and summary of news analyzed.
    
    Args:
        ticker: Stock ticker symbol
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {
                "status": "success",
                "ticker": ticker.upper(),
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "news_count": 0,
                "message": "No recent news found"
            }
        
        # Analyze sentiment for recent news (last 10 articles)
        sentiment_scores = []
        news_analyzed = 0
        
        for article in news[:10]:
            title = article.get('title', '')
            summary = article.get('summary', '')
            # Combine title and summary for better context
            full_text = f"{title}. {summary}" if summary else title
            
            if full_text.strip():
                score = get_sentiment_score(full_text)
                sentiment_scores.append(score)
                news_analyzed += 1
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Determine sentiment label
        if avg_sentiment > 0.2:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.2:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        return {
            "status": "success",
            "ticker": ticker.upper(),
            "sentiment_score": round(avg_sentiment, 3),
            "sentiment_label": sentiment_label,
            "news_count": news_analyzed,
            "message": f"Analyzed {news_analyzed} recent news articles"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error analyzing sentiment: {str(e)}"}


@traceable
def create_stock_agent():
    """
    Create and return a stock trading advisor agent.
    Can be imported and used in other programs.
    """
    return create_agent(
        model=llm,
        tools=[analyze_stock],
        system_prompt="""You are a stock trading advisor. When a user asks about a stock, use the analyze_stock tool to gather comprehensive data, then provide a clear BUY or SELL recommendation.

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

5. **News Sentiment**:
   - Positive sentiment (> 0.2) = Bullish signal, market optimism
   - Negative sentiment (< -0.2) = Bearish signal, market pessimism
   - Neutral sentiment = No strong directional bias from news

   The customer wants to hold the stock for one week so decide whether you think the stock will go up or down in that timeframe.
Always structure your response as:

<Recommendation: BUY> or <Recommendation: SELL>

and only this sentence
""",
    )


# Create the agent instance for import
agent = create_stock_agent()


@traceable
def run_analysis(ticker: str):
    """Run the stock analysis pipeline for a given ticker."""
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Should I buy or sell {ticker}?"
            }
        ]
    })
    return result["messages"][-1].content


# Run the agent
if __name__ == "__main__":

    ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip()

    print(f"\n{'='*60}")
    print(f"Analyzing {ticker.upper()}...")
    print('='*60)
    
    recommendation = run_analysis(ticker)
    print(recommendation)