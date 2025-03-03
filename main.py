import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import google.generativeai as genai
import requests
import json
import os
import re
import PyPDF2
from datetime import datetime, timedelta

custom_css = """
<style>
/* Change background color and font styles */
body {
    background-color: #f4f6f8;
    font-family: 'Segoe UI', sans-serif;
}

/* Header banner styling */
header {
    background: linear-gradient(90deg, #4e73df, #1cc88a);
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
    color: #ffffff;
    text-align: center;
}
header h1 {
    font-size: 2.5rem;
    margin: 0;
}

/* Style for section headers */
h2 {
    color: #333333;
    border-bottom: 2px solid #1cc88a;
    padding-bottom: 4px;
}

/* Style for tabs and expanders */
.stTabs > .css-1d391kg, .css-1d391kg .st-expanderHeader {
    background-color: #ffffff;
    border: 1px solid #dddddd;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Button styling */
div.stButton > button {
    background-color: #1cc88a;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    margin-top: 10px;
}
div.stButton > button:hover {
    background-color: #17a673;
}

/* Additional spacing for markdown */
.stMarkdown {
    margin-top: 10px;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("""
<header>
    <h1>Equity Market Analysis and Stock Prediction</h1>
</header>
""", unsafe_allow_html=True)
# Configuration
NEWS_API_KEY = "2f9f16f2f71b45ffb7b400af499bdb43"
GEMINI_API_KEY = "AIzaSyBqmOTk3yyHcjqwPU3BNiwb57JjcJzT2yc"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Constants and mappings
INDIAN_TICKER_PDF_MAP = {
    "HDFCBANK": "https://www.hdfcbank.com/content/bbp/repositories/723fb80a-2dde-42a3-9793-7ae1be57c87f/?path=/Footer/About%20Us/Investor%20Relation/Detail%20PAges/financial%20results/PDFs/2024/20April/Q4FY24-Earnings-Presentation.pdf",
    "TCS": "https://www.tcs.com/content/dam/tcs/pdf/discover-tcs/investor-relations/management-commentary/ir_presentations/tcs-investor-relations-presentation-4q24.pdf",
    "ADANIENT":"https://www.adanienterprises.com/-/media/Project/Enterprises/Investors/Investor-Downloads/Financial/Q4-FY24.pdf",
    "ADANIPORTS":"https://www.adaniports.com/-/media/Project/Ports/Investor/Investor-Downloads/Quarterly-Results/Results-Q4-FY24.pdf",
    "APOLLOHOSP": "https://www.apollohospitals.com/apollo_pdf/AHEL-Q4-FY23-Earnings-Update.pdf",
    "ASIANPAINT": "https://www.asianpaints.com/content/dam/asianpaints/website/secondary-navigation/investors/financial-results-2/2023-2024/Q4/Q4FY24_Investor_meet_transcript.pdf",
    "AXISBANK": "https://www.axisbank.com/docs/default-source/quarterly-results/press-release-q4fy24.pdf",
    "BAJAJ-AUTO": "https://www.bhil.in/pdf/Press%20Release%20Q4%2023-24%20BHIL%20with%20annexure.pdf",
    "BAJFINANCE": "https://cms-assets.bajajfinserv.in/is/content/bajajfinance/bajaj-finance-q4-conference-call-transcript-updatedpdf?scl=1&fmt=pdf",
    "BAJAJFINSV": "https://cms-assets.bajajfinserv.in/is/content/bajajfinance/bajaj-finance-investor-presentation-q4-fy24-finalpdf?scl=1&fmt=pdf",
    "BEL": "https://bel-india.in/wp-content/uploads/2024/05/BEL-2023-24-Q4-Financial-Results.pdf",
    "BPCL": "https://bel-india.in/wp-content/uploads/2024/05/BEL-2023-24-Q4-Financial-Results.pdf",
    "BHARTIARTL": "https://assets.airtel.in/static-assets/cms/investor/docs/quarterly_results/2023_24/Q4/Published_Results.pdf",
    "BRITANNIA": "https://media.britannia.co.in/Audited_Consolidated_Financial_Results_31_03_2024_15d9ae5e5c.pdf",
    "CIPLA": "https://www.cipla.com/sites/default/files/2024-05/Cipla_Press_Release_10_04_2024_signed.pdf",
    "COALINDIA": "https://nsearchives.nseindia.com/corporate/acctsfinal_02052024181305.pdf",
    "DRREDDY": "https://www.drreddys.com/cms/cms/sites/default/files/2023-05/Earnings%20Release%20-%20Q4%20FY23_1005.pdf",
    "EICHERMOT": "https://eicher.in/content/dam/eicher-motors/investor/financial-and-reports/quarterly-results/Q4-and-FY24-Investor-Presentation.pdf",
    "GRASIM": "https://www.grasim.com/Upload/PDF/grasim-transcript-earnings-call-q4fy22.pdf",
    "HCLTECH": "file:///C:/Users/Aagam/Downloads/HCLTech_Q4_FY24_Investor_Release%20(1).pdf",
    "HDFCLIFE": "https://www.hdfclife.com/content/dam/hdfclifeinsurancecompany/about-us/pdf/investor-relations/financial-information/annual-reports/integrated-annual-report-fy-2023-2024.pdf",
    "INFY": "https://www.infosys.com/investors/reports-filings/quarterly-results/2023-2024/q4/documents/q4-and-12m-fy24-financial-results-auditorsreports.pdf",
    "ITC": "https://www.itcportal.com/investor/pdf/ITC-Quarterly-Result-Presentation-Q4-FY2024.pdf",
    "JSWSTEEL": "",
    "KOTAKBANK": "https://www.kotak.com/content/dam/Kotak/investor-relation/Financial-Result/QuarterlyReport/FY-2024/q4/PressRelease/Q4FY24_Press-Release.pdf",
    "NTPC": "https://ntpc.co.in/sites/default/files/inline-files/TranscriptNTPCQ4FY23.pdf",
    "TATACONSUM": "",
    "TATAMOTORS": "https://static-assets.tatamotors.com/Production/www-tatamotors-com-NEW/wp-content/uploads/2024/05/q4fy24-results-press-release-1.pdf",
    "RELIANCE":"https://rilstaticasset.akamaized.net/sites/default/files/2024-04/22042024-Q4-FY2023-24-Transcript.pdf"
}

# Helper functions for news analysis
def fetch_recent_news(query):
    """Fetch financial news from last 7 days using NewsAPI"""
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&from={one_week_ago}&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"News API Error {response.status_code}: {response.json().get('message', 'Unknown error')}")
            return []
            
        data = response.json()
        return process_articles(data.get('articles', []))
        
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return []

def process_articles(articles):
    """Process and filter news articles"""
    processed = []
    for art in articles:
        content = art.get('content') or art.get('description')
        if not content:
            continue
            
        try:
            processed.append({
                'title': art['title'],
                'content': content[:2000],
                'source': art['source']['name'],
                'date': art['publishedAt'][:10],
                'url': art['url']
            })
        except KeyError:
            continue
            
    return processed[:10]

def analyze_with_gemini(news_articles, query):
    """Perform advanced analysis using Gemini Pro"""
    news_text = "\n\n---\n\n".join([
        f"**Date**: {art['date']}\n**Source**: {art['source']}\n"
        f"**Headline**: {art['title']}\n"
        f"**Content**: {art['content'][:500]}..." 
        for art in news_articles
    ])
    
    prompt = f"""Act as a senior financial analyst. Analyze these recent news articles about {query}:
    {news_text} and just tell me recommended actions (Buy/Hold/Sell) with rationale."""
    
    try:
        with st.spinner('Analyzing with Gemini AI...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"⚠️ Analysis failed: {str(e)}"

# Helper functions for annual report analysis
def contains_word(text: str, word: str) -> bool:
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.search(pattern, text, re.IGNORECASE) is not None

def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text(text, max_length=12000):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_with_gemini_api(prompt):
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
        return "Error processing response"
    except Exception as e:
        return f"API Error: {str(e)}"

def summarize_annual_report(text):
    chunks = split_text(text)
    summaries = []
    
    for chunk in chunks:
        prompt = f"""Analyze this annual report section as a financial expert:
{chunk}
Highlight key metrics, risks, growth opportunities, and financial health indicators.
Use bullet points with specific numbers and percentages."""
        summary = generate_with_gemini_api(prompt)
        summaries.append(summary)
    
    combined = "\n\n".join(summaries)
    final_prompt = f"""Synthesize this annual report analysis into executive summary:
{combined}
Structure with sections: Financial Performance, Risk Factors, Growth Strategy, Valuation Metrics.
Use precise numbers and maintain professional tone."""
    
    return generate_with_gemini_api(final_prompt)

# Main app function
def main():
    st.markdown("---")
    
    # Create tabs in the navigation bar
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Financial Document Processing and Summarization", 
        "Real-Time News Analysis", 
        "Stock Recommendation",
        "Advanced Visualization",
        "Portfolio Customization"
    ])
    
    # Tab 1: Market Intelligence
    with tab1:
        st.header("Annual Report Analysis")
        ticker = st.text_input("Enter NSE ticker symbol:", key="report_input").strip().upper()
        analyze_report = st.button("Analyze Annual Report", key="report_button")
        
        if analyze_report and ticker:
            pdf_url = INDIAN_TICKER_PDF_MAP.get(ticker)
            if not pdf_url:
                st.error("Ticker not found in our database")
                return
                
            with st.spinner("Processing annual report..."):
                try:
                    download_dir = os.path.join("filings", ticker)
                    os.makedirs(download_dir, exist_ok=True)
                    pdf_filename = os.path.join(download_dir, f"{ticker}_report.pdf")
                    download_pdf(pdf_url, pdf_filename)
                    text = extract_text_from_pdf(pdf_filename)
                    
                    if len(text) < 1000:
                        st.error("Insufficient text extracted - may be scanned document")
                        return
                        
                    analysis = summarize_annual_report(text)
                    
                    st.subheader(f"Comprehensive Analysis for {ticker}")
                    st.markdown(analysis)
                    
                    st.download_button(
                        label="Download Full Analysis",
                        data=analysis,
                        file_name=f"{ticker}_annual_analysis.txt"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing report: {str(e)}")

    # Tab 2: Real-Time News Analysis
    with tab2:
        st.header("Real-Time News Analysis")
        stock_name = st.text_input("Enter stock name or symbol:", key="news_input")
        analyze_button = st.button("Get real time news", key="news_button")
        
        if analyze_button and stock_name:
            with st.spinner("Fetching news..."):
                try:
                    prompt = (
                        f"Analyze recent news articles about {stock_name}. "
                        "Provide a summary of key trends, risks, and market sentiment."
                    )
                    response = model.generate_content(prompt)
                    
                    st.subheader(f"News Analysis for {stock_name}")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
    
    # Tab 3: Annual Report Analysis
    with tab3:
        st.header("Market Intelligence Dashboard")
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter stock ticker or index name:", key="intel_input")
        
        if st.button("Analyze Market", key="intel_button") and query:
            with st.spinner('Fetching recent news...'):
                news_articles = fetch_recent_news(query)
                
            if not news_articles:
                st.warning(f"No relevant news found for {query} in past 7 days")
                return
                
            analysis = analyze_with_gemini(news_articles, query)
            
            st.markdown("---")
            st.subheader(f"Market Intelligence Report: {query.upper()}")
            st.markdown(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Articles Analyzed**: {len(news_articles)}")
            st.markdown("---")
            
            st.markdown(analysis, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**Disclaimer**: This AI-generated report is for informational purposes only. Consult a financial advisor before making investment decisions.")
    

    
    # Tab 4: Advanced Visualization
    with tab4:
        st.header("Advanced Visualization Tools")
        
        # Stock selection and date range
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_stock = st.selectbox("Select Stock", options=list(INDIAN_TICKER_PDF_MAP.keys()))
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
            end_date = st.date_input("End Date", value=datetime(2024, 5, 31))
        
        # Generate synthetic price data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        prices = np.random.randn(len(dates)).cumsum() + 100
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        df.set_index('Date', inplace=True)

        # Visualization controls
        with col2:
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Area"])
            show_ma = st.checkbox("Show Moving Averages")
            show_vol = st.checkbox("Show Volume ")

        # Create interactive chart
        fig = go.Figure()
        
        # Price visualization
        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Price'))
        elif chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Price'].shift(1),
                high=df['Price'].rolling(5).max(),
                low=df['Price'].rolling(5).min(),
                close=df['Price'],
                name='Candlestick'
            ))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=df.index, y=df['Price'], fill='tozeroy', name='Price'))

        # Technical indicators
        if show_ma:
            ma_period = st.slider("Moving Average Period", 7, 200, 50)
            df['MA'] = df['Price'].rolling(ma_period).mean()
            fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(dash='dot'), name=f'{ma_period}D MA'))
        
        if show_vol:
            df['Volume'] = np.random.randint(1000, 5000, size=len(df))
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2'))
            fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right'))


        fig.update_layout(
            title=f"{selected_stock} Advanced Analysis",
            xaxis=dict(rangeslider=dict(visible=True)),
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Portfolio Customization
    with tab5:
        st.header("Custom Portfolio Analysis")
        
        with st.expander("Portfolio Settings"):
            portfolio_stocks = st.multiselect(
                "Select Portfolio Stocks",
                options=list(INDIAN_TICKER_PDF_MAP.keys()),
                default=["HDFCBANK", "TCS", "RELIANCE"]
            )
            benchmark = st.selectbox("Select Benchmark Index", ["NIFTY50", "SENSEX", "Custom..."])

        with st.expander("Time Horizon"):
            analysis_period = st.select_slider(
                "Analysis Period",
                options=['1W', '1M', '3M', '6M', '1Y', '3Y', '5Y', 'Custom'],
                value='1Y'
            )
            if analysis_period == 'Custom':
                custom_start = st.date_input("Custom Start Date", value=datetime(2023, 1, 1), key="custom_start")
                custom_end = st.date_input("Custom End Date", value=datetime(2024, 5, 31), key="custom_end")

        with st.expander("Analysis Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                risk_tolerance = st.select_slider(
                    "Risk Tolerance",
                    options=['Low', 'Moderate', 'High', 'Aggressive']
                )
                valuation_method = st.multiselect(
                    "Valuation Methods",
                    ['DCF', 'Comparables', 'Dividend Discount', 'EV/EBITDA'],
                    default=['DCF', 'Comparables']
                )
            with col2:
                technical_indicators = st.multiselect(
                    "Technical Indicators",
                    ['MACD', 'RSI', 'Bollinger Bands', 'Fibonacci Retracement'],
                    default=['MACD', 'RSI']
                )
                scenario_analysis = st.checkbox("Enable Scenario Analysis")

        # Generate custom report
        if st.button("Generate Custom Analysis", key="custom_button"):
            with st.spinner("Building custom analysis..."):
                analysis_params = {
                    "stocks": portfolio_stocks,
                    "benchmark": benchmark,
                    "period": analysis_period,
                    "risk": risk_tolerance,
                    "valuation": valuation_method,
                    "technical": technical_indicators
                }
                
                # Generate analysis report using Gemini
                prompt = f"""Generate a custom equity analysis report with these parameters:
                - Stocks: {portfolio_stocks}
                - Benchmark: {benchmark}
                - Period: {analysis_period}
                - Risk Tolerance: {risk_tolerance}
                - Valuation Methods: {valuation_method}
                - Technical Indicators: {technical_indicators}
                
                Include:
                1. Comparative performance analysis
                2. Risk assessment
                3. Valuation summary
                4. Technical outlook
                5. Portfolio recommendations"""
                
                response = model.generate_content(prompt)
                st.subheader("Custom Analysis Report")
                st.markdown(response.text)
                
                st.download_button(
                    label="Download Portfolio Analysis",
                    data=response.text,
                    file_name="portfolio_analysis.txt"
                )

if __name__ == "__main__":
    main()
