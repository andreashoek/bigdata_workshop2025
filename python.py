import json
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('MASSIVE_API_KEY')

class CompanyAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.massive.com"
    
    def get_company_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get basic company information"""
        url = f"{self.base_url}/v3/reference/tickers/{ticker}?apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_recent_quote(self, ticker: str) -> Dict[str, Any]:
        """Get latest stock price"""
        url = f"{self.base_url}/v3/quotes/stocks/{ticker}?apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get historical stock prices"""
        url = f"{self.base_url}/v3/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_recent_trades(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        """Get recent trading activity"""
        url = f"{self.base_url}/v3/trades/{ticker}?limit={limit}&apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_dividends(self, ticker: str) -> Dict[str, Any]:
        """Get dividend information"""
        url = f"{self.base_url}/v3/reference/dividends?ticker={ticker}&apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """Get financial statements"""
        url = f"{self.base_url}/v3/reference/financials/{ticker}?apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def get_news(self, ticker: str, limit: int = 50) -> Dict[str, Any]:
        """Get recent news articles"""
        url = f"{self.base_url}/v3/reference/news?ticker={ticker}&limit={limit}&apiKey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠ Status {response.status_code}: {response.text[:200]}")
            return {}
    
    def analyze_company(self, ticker: str, include_historical: bool = True) -> Dict[str, Any]:
        """Perform comprehensive company analysis"""
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker.upper()}")
        print(f"{'='*60}\n")
        
        analysis = {}
        
        # Company Fundamentals
        print("Fetching company fundamentals...")
        analysis['fundamentals'] = self.get_company_fundamentals(ticker)
        
        # Current Quote
        print("Fetching current quote...")
        analysis['current_quote'] = self.get_recent_quote(ticker)
        
        # Historical Prices (last 90 days)
        if include_historical:
            print("Fetching historical prices...")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            analysis['historical_prices'] = self.get_historical_prices(ticker, start_date, end_date)
        
        # Recent Trades
        print("Fetching recent trades...")
        analysis['recent_trades'] = self.get_recent_trades(ticker, limit=10)
        
        # Dividends
        print("Fetching dividend data...")
        analysis['dividends'] = self.get_dividends(ticker)
        
        # Financials
        print("Fetching financial statements...")
        analysis['financials'] = self.get_financials(ticker)
        
        # News
        print("Fetching recent news...")
        analysis['news'] = self.get_news(ticker, limit=10)
        
        print("\nAnalysis complete!\n")
        return analysis


def main():
    if not api_key:
        print("Error: MASSIVE_API_KEY not found in environment variables")
        return
    
    analyzer = CompanyAnalyzer(api_key)
    
    # Example: Analyze a company
    ticker = input("Enter company ticker symbol (e.g., AAPL, MSFT, TSLA): ").strip().upper()
    
    if not ticker:
        ticker = "AAPL"  # Default example
        print(f"Using default ticker: {ticker}")
    
    analysis = analyzer.analyze_company(ticker, include_historical=True)
    
    # Save results to file
    output_file = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {ticker}")
    print(f"{'='*60}")
    
    if 'fundamentals' in analysis and analysis['fundamentals']:
        fundamentals = analysis['fundamentals']
        print(f"\nCompany: {fundamentals.get('name', 'N/A')}")
        print(f"Market Cap: ${fundamentals.get('market_cap', 'N/A'):,}" if isinstance(fundamentals.get('market_cap'), (int, float)) else f"Market Cap: N/A")
    
    if 'current_quote' in analysis and analysis['current_quote']:
        quote = analysis['current_quote']
        print(f"\nCurrent Price: ${quote.get('last_price', 'N/A')}")
        print(f"Volume: {quote.get('volume', 'N/A'):,}" if isinstance(quote.get('volume'), (int, float)) else f"Volume: N/A")
    
    if 'news' in analysis and analysis['news']:
        news = analysis['news']
        news_count = len(news.get('results', [])) if isinstance(news, dict) else 0
        print(f"\nRecent News Articles: {news_count}")


if __name__ == "__main__":
    main()