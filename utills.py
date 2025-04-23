import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
from newspaper import Article
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from transformers import pipeline
import re


SENTIMENT_ANALYSIS_MODEL = (
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

sentiment_analyzer = pipeline(
    "sentiment-analysis", model=SENTIMENT_ANALYSIS_MODEL
)

# artifact_sentiment_path = "./artifacts/sentiment"
# sentiment_analyzer = pipeline(
#     "sentiment-analysis",
#     model=artifact_sentiment_path,
#     tokenizer=artifact_sentiment_path
# )

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# artifact_summarizer_path = "./artifacts/summarization"
# summarizer = pipeline(
#     "summarization",
#     model=artifact_summarizer_path,
#     tokenizer=artifact_summarizer_path
# )



def scrape_google_news(query, max_pages=3):
    """
    Scrape Google News search results sorted by most recent.
    :param query:      Search term
    :param max_pages:  How many pages of 10 results to fetch (default 3 → up to 30)
    :return:           List of articles dicts
    """
    articles = []
    q = quote_plus(query)
    for page in range(max_pages):
        start = page * 10
        url = (
            "https://www.google.com/search"
            f"?q={q}+stock"  # Search term :contentReference[oaicite:1]{index=1}
            "&tbm=nws"                    # News tab :contentReference[oaicite:4]{index=4}
            "&tbs=qdr:d,sbd:1"            # Past day + sort by date :contentReference[oaicite:5]{index=5}
            f"&start={start}"             # Pagination :contentReference[oaicite:6]{index=6}
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9"
        }

        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Error: status {resp.status_code}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        for el in soup.select("div.SoaBEf"):
            title_el   = el.select_one("div.MBeuO")
            link_el    = el.find("a", href=True)
            source_el  = el.select_one(".NUnG9d span")
            snippet_el = el.select_one(".GI74Re")
            date_el    = el.select_one(".LfVVr")

            articles.append({
                "title":     title_el.get_text(strip=True) if title_el else None,
                "link":      link_el["href"] if link_el else None,
                "source":    source_el.get_text(strip=True) if source_el else None,
                "snippet":   snippet_el.get_text(strip=True) if snippet_el else None,
                "published": date_el.get_text(strip=True) if date_el else None
            })
        # If fewer than 10 results on this page, stop early
        if len(soup.select("div.SoaBEf")) < 10:
            break

    return articles


def fix_ocr_spacing(text):
    # Remove lines or fragments containing unwanted phrases
    blacklist_phrases = [
        "Identify", "SPECIFIC", "stock implications", "from this news","news"
        "negative", "positive", "neutral"
    ]

    # Lowercase version of the text for matching
    lower_text = text.lower()

    # Skip if any blacklist phrase is fully present in the text
    if any(phrase.lower() in lower_text for phrase in blacklist_phrases):
        return ""  # Or return None if you want to ignore this entry entirely

    # Merge single letters like: 'p r i c e' => 'price'
    text = re.sub(r'\b(?:[a-zA-Z]\s){2,}[a-zA-Z]\b', lambda m: m.group().replace(" ", ""), text)

    # Fix common misplaced punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)

    return text.strip()

def get_article_content(url):
    # url = decode_url(url)

    article = Article(url, language='en')

    # Override headers to mimic a browser
    article.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0 Safari/537.36'
    }

    article.download()
    article.parse()
    return article.text


def fetch_final_dict(news, ticker):
    article_data = []

    for art in tqdm(news):
        content = ''
        try:
            content = get_article_content(art['link'])
        except Exception as e:
            print(f"❌ Failed to fetch content for {art['link']}: {e}")


        article_data.append({
            'symbol':ticker,
            'published': art['published'],
            'title': art['title'],
            'link': art['link'],
            'source': art['source'],
            'content': content,
            'sentiment': sentiment_analyzer(content[:1024])
        })
    return pd.DataFrame(article_data)


def get_article_df(ticker,max_pages):

    news = scrape_google_news(ticker, max_pages=max_pages)
    df = fetch_final_dict(news, ticker)

    df['sentiment_label'] = df['sentiment'].apply(
        lambda lst: lst[0]['label'] if isinstance(lst, list) and lst else None
    )
    df['softmax_score'] = df['sentiment'].apply(
        lambda lst: lst[0]['score'] if isinstance(lst, list) and lst else None
    )
    df = df.drop(columns=['sentiment'])
    df = df.drop_duplicates(subset=['title'], keep='first')[df['content'] != '']

    return df


class predictor:
    def __init__(self, data, look_back=1):
        self.data = data.copy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.look_back = look_back
        self.scaled_data = None
        self.X_train = None
        self.y_train = None

    def prepare_data(self):
        delta = self.data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        self.data['RSI'] = 100 - (100 / (1 + rs)).fillna(0)

        features = np.column_stack([
            self.data['Open'],
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            self.data['RSI']
        ])
        self.scaled_data = self.scaler.fit_transform(features)

        def create_sequences(dataset, look_back):
            X, y = [], []
            for i in range(len(dataset) - look_back):
                X.append(dataset[i:i + look_back])
                y.append(dataset[i + look_back][3])
            return np.array(X), np.array(y)

        self.X_train, self.y_train = create_sequences(self.scaled_data, self.look_back)
        return self.X_train, self.y_train

    def train_lstm(self, epochs=5, batch_size=1):
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def make_predictions(self, days=5):
        if self.X_train is None:
            self.prepare_data()

        predictions = self.model.predict(self.X_train)
        inv_preds = []
        for i in range(self.look_back, len(self.scaled_data)):
            row = self.scaled_data[i].copy()
            row[3] = predictions[i - self.look_back]
            inv = self.scaler.inverse_transform([row])[0][3]
            inv_preds.append(inv)

        return inv_preds[-days:], self.data['Close'].values[-days:]

    def forecast_future(self, days=5):
        if self.scaled_data is None:
            self.prepare_data()

        input_seq = self.scaled_data[-self.look_back:].copy()
        future_predictions = []

        for _ in range(days):
            input_reshaped = input_seq.reshape(1, self.look_back, self.scaled_data.shape[1])
            pred_scaled_close = self.model.predict(input_reshaped)[0][0]

            next_row = input_seq[-1].copy()
            next_row[3] = pred_scaled_close
            future_predictions.append(next_row.copy())

            input_seq = np.append(input_seq[1:], [next_row], axis=0)

        inv_preds = [self.scaler.inverse_transform([p])[0][3] for p in future_predictions]
        return inv_preds



def generate_finance_summary(text, sentiment):
    """Generate a fast summary using T5-small"""
    try:
        prompt = f"""
                    Identify SPECIFIC stock implications from this news. 
                    News: {text} in {sentiment}
                  """
        chunk_size = 400  # Keep chunks short for T5-small speed
        chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]

        # Use first 2 chunks max
        result = summarizer(
            " ".join(chunks[:2]),
            max_length=80,
            min_length=20,
            do_sample=False
        )

        # return f"{sentiment.capitalize()} - {result[0]['summary_text']}"
        return fix_ocr_spacing(f"{result[0]['summary_text']}")


    
    except Exception as e:
        return f"Summary error: {str(e)}"



def train_and_predict(ticker,look_back=1, days=5):
    # Fetch historical data
    data = yf.Ticker(ticker).history(period="1y")
    data.dropna().reset_index(inplace=True)

    # Initialize predictor
    model = predictor(data, look_back=look_back)
    model.prepare_data()
    # Train LSTM model
    model.train_lstm(epochs=10, batch_size=16)

    # Make predictions for the next 'days' days
    future_prices = model.forecast_future(days=days)
    future_prices = [round(price, 3) for price in future_prices]
    return future_prices



def get_technical_indicators(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="3mo", interval="1d")
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    delta = df["Close"].diff()
    gain  = delta.where(delta>0, 0)
    loss  = -delta.where(delta<0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df.dropna()

# ————————————————————————————————
# 5) RECOMMENDATION LOGIC
# ————————————————————————————————

import pandas as pd


def recommend(symbol, tech, forecast, sent, hold):
    """
    Generate a recommendation DataFrame for a given stock based on:
      - Technical indicators (tech DataFrame)
      - Forecast list of future prices
      - Sentiment data (sent: dict with 'counts', 'summaries', 'news')
      - Holding details (hold: dict with 'quantity', 'buy_price')
    """
    # 1) Position P/L
    close_price = float(tech["Close"].iloc[-1])
    quantity    = hold.get("quantity", 0)
    buy_price   = hold.get("buy_price", 0)
    invested_value = buy_price * quantity
    current_value  = close_price * quantity
    gain_loss_pct  = ((current_value - invested_value) / invested_value * 100) if invested_value else 0
    # profit_pct     = ((close_price - buy_price) / buy_price * 100) if buy_price else 0

    # 2) Forecast trend
    forecast_trend = "Neutral"
    if isinstance(forecast, list) and len(forecast) >= 2:
        try:
            change_pct = ((forecast[-1] - forecast[0]) / forecast[0]) * 100 if forecast[0] else 0
            if change_pct > 1.5:
                forecast_trend = "Upward"
            elif change_pct < -1.5:
                forecast_trend = "Downward"
        except Exception:
            pass

    # 3) Technical signals
    rsi = float(tech.get("RSI", pd.Series([0])).iloc[-1])
    macd_line = float(tech.get("MACD", pd.Series([0])).iloc[-1])
    # 3-day SMA of MACD (fallback to last MACD if not enough points)
    macd_signal_val = tech.get("MACD", pd.Series([macd_line]))
    signal_line = macd_signal_val.iloc[-4:-1].mean() if len(macd_signal_val) >= 4 else macd_line
    macd_signal = "Bullish" if macd_line > signal_line else "Bearish"

    # 4) Support/resistance & stop loss
    support    = float(tech.get('Low', pd.Series([0])).rolling(5).mean().iloc[-1])
    resistance = float(tech.get('High', pd.Series([0])).rolling(5).mean().iloc[-1])
    stop_loss  = support * 0.97

    # 5) Sentiment analysis
    counts    = sent.get('counts', {}) or {}
    total_news = sum(counts.values())
    pos_ratio = counts.get('positive', 0) / total_news if total_news else 0
    neg_ratio = counts.get('negative', 0) / total_news if total_news else 0

    macd_vals       = tech.get("MACD", pd.Series([0.0]))
    macd_line       = float(macd_vals.iloc[-1])
    signal_line     = macd_vals.iloc[-4:-1].mean() if len(macd_vals) >= 4 else macd_line
    macd_bullish    = macd_line > signal_line

    # Determine recommendation
    recommendation = "Hold"
    reasoning      = []

    # If forecast is downward:
    if forecast_trend == "Downward":
        if gain_loss_pct > 0 or neg_ratio > 0.5:
            recommendation = "Sell"
            reasoning.append("Lock in gains or cut losses as outlook is negative")
        else:
            recommendation = "Hold"
            reasoning.append("Waiting for positive signal before exiting")

    # If forecast is upward:
    elif forecast_trend == "Upward":
        if gain_loss_pct > 10:
            recommendation = "Sell"
            reasoning.append("Take profits with strong upward outlook")
        elif gain_loss_pct > 0:
            recommendation = "Hold"
            reasoning.append("Moderate gains; let position run")
        else:
            recommendation = "Hold"
            reasoning.append("Currently at a loss but forecast is bullish—hold through recovery")

    # Neutral forecast:
    else:
        if pos_ratio > 0.6 and gain_loss_pct > 0:
            recommendation = "Accumulate"
            reasoning.append("Positive sentiment and modest gains—build position")
        elif pos_ratio > 0.6 and gain_loss_pct <= 0:
            recommendation = "Buy"
            reasoning.append("Positive sentiment; buying opportunity at a dip")
        elif macd_bullish and gain_loss_pct < 0:
            recommendation = "Buy"
            reasoning.append("Bullish MACD and current dip—add to position")
        else:
            recommendation = "Hold"
            reasoning.append("No clear signal; maintain current position")

    
    reasoning_text = " . ".join(reasoning) if reasoning else "Neutral market conditions"

    # 7) Top headlines
    # news_df = sent.get('news')
    # top_headlines = []
    # if isinstance(news_df, pd.DataFrame) and not news_df.empty:
    #     top_headlines = news_df.head(3)[['title', 'published']].to_dict('records')


    # 8) Return DataFrame
    return pd.DataFrame([{
        "Stock":          symbol,
        "Quantity":       quantity,
        "Buy Price":      buy_price,
        "Current Price":  close_price,
        "Invested Value": invested_value,
        "Current Value":  current_value,
        "RSI (14)":       rsi,
        "MACD Signal":    macd_signal,
        "Support":        support,
        "Resistance":     resistance,
        "Stop Loss":      stop_loss,
        "Forecast Trend": forecast_trend,
        "Recommendation": recommendation,
        "Reasoning":      reasoning_text,
        # "Key Headlines":  top_headlines
    }])
