import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from plotly.subplots import make_subplots

from utills import (
    get_article_df,
    get_technical_indicators,
    train_and_predict,
    recommend,
    generate_finance_summary
)

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Advisor")

# --- Session State Management ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'analysis' not in st.session_state:
    st.session_state.analysis = {}

# --- Sidebar ---
with st.sidebar:
    st.header("📥 Portfolio Manager")

    # Portfolio entry form
    # with st.sidebar.form("entry_form", clear_on_submit=True):
    #     symbol = st.text_input("Stock Symbol", value="ELV").upper()
    #     quantity = st.number_input("Quantity", min_value=1, value=100)
    #     buy_price = st.number_input("Buy Price", min_value=0.01, value=300.0)
    #     add = st.form_submit_button("➕ Add Stock")
    #     if add and symbol:
    #         st.session_state.portfolio[symbol] = {"quantity": quantity, "buy_price": buy_price}
    #         st.success(f"Added {symbol}: {quantity} shares @ ${buy_price:.2f}")

    with st.sidebar.form("entry_form", clear_on_submit=True):
        symbol = st.text_input("Stock Symbol", placeholder="Enter stock ticker (e.g., AAPL)").upper()
        quantity = st.number_input("Quantity", min_value=1, placeholder="Enter number of shares")
        buy_price = st.number_input("Buy Price", min_value=0.01, placeholder="Enter buy price per share")
        add = st.form_submit_button("➕ Add Stock")
        if add and symbol:
            st.session_state.portfolio[symbol] = {"quantity": quantity, "buy_price": buy_price}
            st.success(f"Added {symbol}: {quantity} shares @ ${buy_price:.2f}")

    # Analyze and Clear buttons
    enable_analyze = st.sidebar.button("🔍 Analyze Portfolio")
    enable_clear = st.sidebar.button("🗑️ Clear Portfolio")
    if enable_clear:
        st.session_state.portfolio.clear()
        st.session_state.run_analysis = False

    # Show current holdings
    if st.session_state.portfolio:
        st.sidebar.subheader("Current Holdings")
        for sym, h in st.session_state.portfolio.items():
            st.sidebar.write(f"**{sym}**: {h['quantity']} @ ${h['buy_price']}")
    else:
        st.sidebar.info("No holdings. Add stocks above.")

# --- Main Page ---
st.title("📈 AI Stock Advisor")

if not st.session_state.portfolio:
    st.info("Add stocks using the sidebar to begin analysis")
    st.stop()

# Trigger analysis
if enable_analyze:
    st.session_state.run_analysis = True
    st.session_state.analysis = {}
    status_box = st.empty()

    with st.spinner("Performing analysis..."):
        for sym, hold in st.session_state.portfolio.items():
            # Prepare defaults
            tech = None
            forecast = None
            news = None
            counts = {}
            summaries = {}

            try:
                status_box.info(f"🔄 [{sym}] Calculating technical indicators...")
                tech = get_technical_indicators(sym)

                status_box.info(f"📈 [{sym}] Forecasting stock prices...")
                forecast = train_and_predict(sym, look_back=5)

                status_box.info(f"📰 [{sym}] Fetching recent news articles...")
                news = get_article_df(sym, max_pages=2)

                status_box.info(f"🧠 [{sym}] Generating sentiment summaries...")
                counts = news['sentiment_label'].value_counts().to_dict()
                summaries = {
                    lbl: generate_finance_summary(
                        news[news.sentiment_label == lbl]['content'].str.cat(sep='. '),
                        lbl
                    ) for lbl in ['positive', 'negative', 'neutral']
                }

                status_box.info(f"🤖 [{sym}] Generating final recommendation...")
                rec_df = recommend(
                    sym,
                    tech,
                    forecast,
                    {'counts': counts, 'summaries': summaries, 'news': news},
                    hold
                )
            except Exception as e:
                status_box.error(f"❌ [{sym}] Analysis failed: {e}")
                rec_df = pd.DataFrame([{}])
            finally:
                # Ensure we always have a dict for recommendation
                rec_dict = rec_df.iloc[0].to_dict() if not rec_df.empty else {}
                st.session_state.analysis[sym] = {
                    'tech': tech,
                    'forecast': forecast,
                    'news': news,
                    'counts': counts,
                    'summaries': summaries,
                    'recommendation': rec_dict
                }
                status_box.success(f"✅ [{sym}] Analysis complete.")
        status_box.success("🎉 All stock analyses completed successfully.")

def show_technical_glossary():
    with st.expander("📚 Technical Terms Glossary", expanded=False):
        st.markdown("""
        - **RSI (Relative Strength Index):** Momentum indicator (30=oversold, 70=overbought)
        - **MACD:** Trend-following indicator showing relationship between two moving averages
        - **Support:** Price level where buying interest is expected to emerge
        - **Resistance:** Price level where selling pressure may increase
        - **Stop Loss:** Automatic sell trigger to limit potential losses
        """)

# --- Tabs ---
tab1, tab2 = st.tabs(["Recommendations", "Charts"])

with tab1:
    for sym, analysis in st.session_state.analysis.items():
        rec = analysis.get('recommendation', {})
        if not rec:
            st.warning(f"No recommendation generated for {sym}.")
            continue

        with st.container():
            st.subheader(f"📌 {sym}")

            with st.expander("📈 Recommendation", expanded=True):
                reco = rec.get('Recommendation', 'Hold')
                color = "green" if reco in ["Buy", "Accumulate"] else "red" if "Sell" in reco else "blue"
                st.markdown(f"### <span style='color:{color}'>{reco}</span>", unsafe_allow_html=True)

                if reco in ["Buy", "Accumulate"]:
                    st.write(analysis['summaries']['positive'])
                elif "Sell" in reco:
                    st.write(analysis['summaries']['negative'])
                else:
                    st.markdown("##### Positive Catalysts:")
                    st.write(analysis['summaries']['positive'])
                    st.markdown("##### Negative Briefs:")
                    st.write(analysis['summaries']['negative'])

                st.markdown(f"**Rationale:** {rec.get('Reasoning', 'Neutral market conditions')}")
                rsi_val = rec.get('RSI (14)', None)
                if rsi_val is not None:
                    progress = (rsi_val - 30) / 40
                    st.progress(progress, text=f"RSI: {rsi_val:.1f} ({progress:.0%} to overbought)")

                st.markdown(f"**Forecast Trend:** {rec.get('Forecast Trend')}")
                

            st.markdown("---")
            # Metrics grid
            current_price = rec.get('Current Price', 0)
            buy_price = rec.get('Buy Price', 0)
            delta_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price else 0
            current_value = rec.get('Current Value', 0)
            invested_value = rec.get('Invested Value', 0)
            profit_loss = current_value - invested_value

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}", f"{delta_pct:.2f}%")
            col2.metric("Current Value", f"${current_value:.2f}", f"${profit_loss:.2f}")
            col3.metric("Quantity", rec.get('Quantity', 0))

            col4, col5, col6 = st.columns(3)
            col4.metric("MACD Signal", rec.get('MACD Signal', 'N/A'))
            col5.metric("RSI", f"{rec.get('RSI (14)', 0):.1f}")
            col6.metric("Support", f"${rec.get('Support', 0):.2f}")

            next_price = analysis['forecast'][0] if analysis['forecast'] else 0
            stop_loss = rec.get('Stop Loss', 0)
            rsi_val = rec.get('RSI (14)', 50)
            risk_level = 'High' if rsi_val > 70 else 'Medium' if rsi_val > 45 else 'Low'

            col7, col8, col9 = st.columns(3)
            col7.metric("Next Price", f"${next_price:.2f}", f"${next_price - current_price:.2f}", delta_color='normal')
            col8.metric("Stop Loss", f"${stop_loss:.2f}", delta="3% below support")
            col9.metric("Risk Level", risk_level)

            show_technical_glossary()
            st.markdown("----")

            # Display news articles
            with st.expander(f"📰 Recent News Articles", expanded=True):
                news = analysis.get('news', pd.DataFrame()).head(5)
                if not news.empty:
                    for _, row in news.iterrows():
                        st.markdown(f"**{row['title']}**")
                        st.write(f"Published {row['published']}")
                        st.write(f"[Read more]({row['link']})")
                        # st.markdown("---")
                else:
                    st.warning("No news articles found.")

with tab2:
    st.header("Price Charts")
    if not st.session_state.analysis:
        st.warning("Run analysis first")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for sym in st.session_state.portfolio:
            analysis = st.session_state.analysis.get(sym, {})
            forecast = analysis.get('forecast', [])
            if not isinstance(forecast, list) or not forecast:
                continue

            forecast_vals = [float(v) for v in forecast if pd.notna(v)]
            try:
                stock_data = yf.download(sym, period="7d", interval="1d")
                last_date = stock_data.index[-1]
            except Exception as e:
                st.error(f"Could not fetch data for {sym}: {e}")
                continue

            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_vals))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_vals,
                mode='lines+markers+text',
                name=sym,
                text=[f"${v:.2f}" for v in forecast_vals],
                textposition="top center",
                marker=dict(symbol="circle")
            ))

        fig.update_layout(title="Forecasted Prices (Next 5 Days)",
                          xaxis_title="Date", yaxis_title="Price ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_hist = go.Figure()
        for sym in st.session_state.portfolio:
            try:
                hist = yf.Ticker(sym).history(period='max')
                if 'Close' not in hist.columns or hist['Close'].dropna().empty:
                    st.warning(f"No valid 'Close' data for {sym}. Skipping.")
                    continue
                fig_hist.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name=sym, mode='lines'))
            except Exception as e:
                st.error(f"Could not fetch historical data for {sym}: {e}")

        fig_hist.update_layout(title="Historical Close Prices",
                               xaxis_title="Date", yaxis_title="Close Price ($)", height=500)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.header("Portfolio Composition")
    col1, col2 = st.columns(2)
    with col1:
        try:
            alloc = {
                sym: h['quantity'] * st.session_state.analysis[sym]['recommendation'].get('Current Price', 0)
                for sym, h in st.session_state.portfolio.items()
            }
            pie = px.pie(names=list(alloc.keys()), values=list(alloc.values()),
                         title="Portfolio Value Distribution")
            st.plotly_chart(pie, use_container_width=True)
        except KeyError as e:
            st.error(f"Missing data for allocation chart: {e}")

    with col2:
        try:
            sentiment_data = []
            for sym, analysis in st.session_state.analysis.items():
                for sent, cnt in analysis.get('counts', {}).items():
                    sentiment_data.append({'Stock': sym, 'Sentiment': sent.capitalize(), 'Count': cnt})
            df_sent = pd.DataFrame(sentiment_data)
            sunburst = px.sunburst(df_sent, path=['Stock', 'Sentiment'], values='Count',
                                   title="News Sentiment Distribution")
            st.plotly_chart(sunburst, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating sentiment chart: {e}")
