# Stock Trading Agent

A simple AI-powered stock trading advisor using LangChain and DeepSeek.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API key:
```
DEEPSEEK_API_KEY=your_key_here
```

## Usage

### Run the CLI:
```bash
python stock_agent.py
```

### Use in your code:
```python
from stock_agent import agent

result = agent.invoke({
    "messages": [{"role": "user", "content": "Should I buy or sell AAPL?"}]
})
print(result["messages"][-1].content)
```

filter for <Recommendation: BUY** or **Recommendation: SELL>