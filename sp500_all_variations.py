#!/usr/bin/env python3
"""
S&P 500 All Variations Monitor - Complete Market Overview
Tracks ALL stock movements with sector breakdowns and market summary.
Runs alongside the main monitor for comprehensive coverage.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import aiohttp
from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_ALL_VARIATIONS")
LANGUAGE = os.getenv("LANGUAGE", "BOTH").upper()  # EN, FR, or BOTH
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

COMPANY_NAMES = {}  # Will store ticker -> company name mapping
PREVIOUS_PRICES = {}  # Prices from last update (30 min ago)
OPEN_PRICES = {}  # Prices from market open
YESTERDAY_CLOSES = {}  # Prices from yesterday close


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sector classifications for S&P 500
SECTOR_KEYWORDS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'AVGO', 'QCOM', 'TXN', 'ORCL'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY', 'AMGN', 'CVS', 'CI'],
    'Consumer': ['AMZN', 'TSLA', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'COST'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],
    'Industrial': ['BA', 'HON', 'UNP', 'CAT', 'RTX', 'GE', 'LMT', 'DE', 'MMM', 'UPS'],
}


# ============================================================================
# TICKER SCRAPING
# ============================================================================

def to_ticker_symbol(sym: str) -> str:
    """Normalize ticker to Yahoo Finance format."""
    return sym.replace(".", "-").upper().strip()

def get_company_names() -> Dict[str, str]:
    """
    Fetch company names from Wikipedia S&P 500 table.
    Returns dict: {ticker: company_name}
    """
    logger.info("Fetching company names from Wikipedia...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns and "Security" in tbl.columns:
                mapping = {}
                for _, row in tbl.iterrows():
                    ticker = str(row["Symbol"]).replace(".", "-").upper().strip()
                    company = str(row["Security"]).strip()
                    mapping[ticker] = company
                
                logger.info(f"✓ Loaded {len(mapping)} company names")
                return mapping
        
        return {}
    except Exception as e:
        logger.error(f"Error fetching company names: {e}")
        return {}

def track_price_history(market_data: Dict):
    """
    Track prices across different timeframes.
    Called after fetching market data each run.
    """
    global PREVIOUS_PRICES, OPEN_PRICES, YESTERDAY_CLOSES
    
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # Check if this is market open (9:30 AM)
    # In EDT: 13:30 UTC, in EST: 14:30 UTC
    is_market_open = (current_hour == 13 and current_minute == 30) or \
                     (current_hour == 14 and current_minute == 30)
    
    for ticker, data in market_data.items():
        current_price = data['current_price']
        
        # At market open, store opening prices
        if is_market_open or ticker not in OPEN_PRICES:
            OPEN_PRICES[ticker] = data['open']
            
            # Store yesterday's close (approximate with today's open if gap exists)
            if ticker not in YESTERDAY_CLOSES:
                YESTERDAY_CLOSES[ticker] = data['open']
        
        # Always update previous price for next run
        PREVIOUS_PRICES[ticker] = current_price
        

def get_sp500_tickers() -> List[str]:
    """Scrape Wikipedia for S&P 500 ticker list."""
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns:
                tickers = sorted({to_ticker_symbol(s) for s in tbl["Symbol"].astype(str)})
                logger.info(f"✓ Loaded {len(tickers)} tickers")
                return tickers
        
        raise RuntimeError("Couldn't parse tickers from Wikipedia")
    
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        raise


def format_enhanced_ticker(ticker: str, data: Dict) -> str:
    """
    Format ticker with company name and multi-timeframe changes.
    
    Returns: "Company Name | TICKER: +X.XX% (+Y.YY% since last | +Z.ZZ% since yesterday)"
    """
    company = COMPANY_NAMES.get(ticker, ticker)
    current = data['current_price']
    
    # Change since market open
    open_price = OPEN_PRICES.get(ticker, data['open'])
    if pd.notna(open_price) and open_price > 0:  # ✅ Added validation
        since_open = ((current - open_price) / open_price) * 100
    else:
        since_open = 0.0
    
    # Change since last update (30 min ago)
    if ticker in PREVIOUS_PRICES:
        prev = PREVIOUS_PRICES[ticker]
        since_last = ((current - prev) / prev) * 100
    else:
        since_last = 0.0
    
    # Change since yesterday close
    yesterday = YESTERDAY_CLOSES.get(ticker, open_price)
    since_yesterday = ((current - yesterday) / yesterday) * 100
    
    # Build formatted string
    result = f"**{company} | {ticker}**: {since_open:+.2f}% "
    result += f"({since_last:+.2f}% since last | {since_yesterday:+.2f}% since yesterday close)"
    
    return result


# ============================================================================
# DATA FETCHING
# ============================================================================

async def fetch_all_market_data(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch current day's market data for ALL tickers.
    Returns comprehensive data for every stock.
    """
    logger.info(f"Fetching complete market data for {len(tickers)} tickers...")
    
    results = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        
        try:
            data = yf.download(
                batch,
                period="1d",
                interval="1m",
                group_by='ticker',
                progress=False,
                threads=True
            )
            
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]
                    
                    if ticker_data.empty:
                        continue
                    
                    open_price = ticker_data['Open'].iloc[0]
                    current_price = ticker_data['Close'].iloc[-1]
                    high = ticker_data['High'].max()
                    low = ticker_data['Low'].min()
                    volume = ticker_data['Volume'].sum()
                    
                    # Skip if any value is NaN or invalid
                    if pd.isna(open_price) or pd.isna(current_price) or open_price == 0:
                        continue
                    
                    change_pct = ((current_price - open_price) / open_price) * 100
                    
                    # Skip if change is NaN or infinite
                    if pd.isna(change_pct) or np.isinf(change_pct):
                        continue
                    
                    results[ticker] = {
                        'current_price': float(current_price),
                        'open': float(open_price),
                        'high': float(high),
                        'low': float(low),
                        'volume': int(volume),
                        'change_pct': float(change_pct)
                    }
                
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Batch fetch error: {e}")
            continue
        
        if i + BATCH_SIZE < len(tickers):
            await asyncio.sleep(0.5)
    
    logger.info(f"✓ Fetched data for {len(results)} stocks")
    return results


# ============================================================================
# MARKET ANALYSIS
# ============================================================================

def calculate_market_statistics(market_data: Dict[str, Dict]) -> Dict:
    """Calculate comprehensive market statistics."""
    if not market_data:
        return {}
    
    changes = [data['change_pct'] for data in market_data.values()]
    volumes = [data['volume'] for data in market_data.values()]
    
    stats = {
        'total_stocks': len(market_data),
        'gainers': sum(1 for c in changes if c > 0),
        'losers': sum(1 for c in changes if c < 0),
        'unchanged': sum(1 for c in changes if c == 0),
        'avg_change': float(np.mean(changes)),
        'median_change': float(np.median(changes)),
        'std_dev': float(np.std(changes)),
        'max_gain': float(max(changes)),
        'max_loss': float(min(changes)),
        'total_volume': sum(volumes),
        'avg_volume': float(np.mean(volumes))
    }
    
    # Market sentiment
    if stats['avg_change'] > 0.5:
        stats['sentiment'] = 'Strongly Bullish 🟢'
    elif stats['avg_change'] > 0:
        stats['sentiment'] = 'Bullish 🟢'
    elif stats['avg_change'] > -0.5:
        stats['sentiment'] = 'Bearish 🔴'
    else:
        stats['sentiment'] = 'Strongly Bearish 🔴'
    
    return stats


def explain_volatility_and_sentiment(stats: Dict) -> str:
    """
    Enhanced volatility and sentiment explanation.
    """
    std_dev = stats.get('std_dev', 0)
    avg_change = stats.get('avg_change', 0)
    gainers = stats.get('gainers', 0)
    total = stats.get('total_stocks', 1)
    
    explanation = "\n**📊 MARKET ANALYSIS**\n"
    
    # Volatility classification
    if std_dev < 0.5:
        vol_desc = "😴 Very Low - Market sleeping"
        strategy = "Wait for better setups"
    elif std_dev < 1.0:
        vol_desc = "🟢 Low - Calm conditions"
        strategy = "Good for trend following"
    elif std_dev < 1.5:
        vol_desc = "🔵 Normal - Healthy market"
        strategy = "Standard strategies work"
    elif std_dev < 2.5:
        vol_desc = "🟡 Elevated - Larger swings"
        strategy = "Widen stops, reduce size"
    else:
        vol_desc = "🔴 Extreme - High risk"
        strategy = "Reduce exposure significantly"
    
    explanation += f"**Volatility**: {vol_desc} ({std_dev:.2f}%)\n"
    explanation += f"**Strategy**: {strategy}\n\n"
    
    # Sentiment with breadth
    gainer_pct = (gainers / total) * 100
    
    if gainer_pct > 70:
        sentiment_desc = "🟢 Very Strong - Broad rally"
    elif gainer_pct > 60:
        sentiment_desc = "🟢 Strong - Healthy breadth"
    elif gainer_pct > 50:
        sentiment_desc = "🔵 Neutral-Bullish"
    elif gainer_pct > 40:
        sentiment_desc = "🟡 Neutral-Bearish"
    elif gainer_pct > 30:
        sentiment_desc = "🔴 Weak - Selling pressure"
    else:
        sentiment_desc = "🚨 Very Weak - Capitulation risk"
    
    explanation += f"**Market Sentiment**: {sentiment_desc}\n"
    explanation += f"**Breadth**: {gainers} gainers ({gainer_pct:.1f}%)\n"
    
    return explanation



def analyze_by_sector(market_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Analyze performance by sector."""
    sector_performance = {}
    
    for sector, tickers in SECTOR_KEYWORDS.items():
        sector_data = [
            market_data[ticker]['change_pct'] 
            for ticker in tickers 
            if ticker in market_data
        ]
        
        if sector_data:
            sector_performance[sector] = {
                'avg_change': float(np.mean(sector_data)),
                'stocks_tracked': len(sector_data),
                'best': float(max(sector_data)),
                'worst': float(min(sector_data))
            }
    
    # Sort by performance
    sector_performance = dict(
        sorted(sector_performance.items(), key=lambda x: x[1]['avg_change'], reverse=True)
    )
    
    return sector_performance


def get_top_performers(market_data: Dict[str, Dict], n: int = 10) -> Tuple[List, List]:
    """Get top N gainers and losers."""
    sorted_by_change = sorted(
        market_data.items(),
        key=lambda x: x[1]['change_pct'],
        reverse=True
    )
    
    top_gainers = [(ticker, data['change_pct']) for ticker, data in sorted_by_change[:n]]
    top_losers = [(ticker, data['change_pct']) for ticker, data in sorted_by_change[-n:]]
    top_losers.reverse()
    
    return top_gainers, top_losers


def get_volatility_leaders(market_data: Dict[str, Dict], n: int = 5) -> List[Tuple]:
    """Get stocks with highest intraday volatility."""
    volatility = []
    
    for ticker, data in market_data.items():
        daily_range = ((data['high'] - data['low']) / data['open']) * 100
        volatility.append((ticker, daily_range, data['change_pct']))
    
    volatility.sort(key=lambda x: x[1], reverse=True)
    return volatility[:n]




def get_volume_leaders(market_data: Dict[str, Dict], n: int = 5) -> List[Tuple]:
    """Get stocks with highest volume."""
    volume_data = [
        (ticker, data['volume'], data['change_pct']) 
        for ticker, data in market_data.items()
    ]
    
    volume_data.sort(key=lambda x: x[1], reverse=True)
    return volume_data[:n]


# ============================================================================
# DISCORD NOTIFICATION
# ============================================================================

async def send_discord_message(content: str) -> bool:
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.error("DISCORD_WEBHOOK_ALL_VARIATIONS not set")
        return False
    
    # Split message if too long (Discord 2000 char limit)
    messages = []
    if len(content) > 2000:
        parts = content.split('\n\n')
        current_message = ""
        
        for part in parts:
            if len(current_message) + len(part) + 2 < 2000:
                current_message += part + "\n\n"
            else:
                messages.append(current_message)
                current_message = part + "\n\n"
        
        if current_message:
            messages.append(current_message)
    else:
        messages = [content]
    
    # Send all message parts
    for i, msg in enumerate(messages):
        payload = {
            "content": msg,
            "username": "SP500 All Variations",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/3658/3658773.png"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DISCORD_WEBHOOK_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 204:
                        logger.info(f"✓ Discord message part {i+1}/{len(messages)} sent")
                    else:
                        error = await response.text()
                        logger.error(f"Discord error {response.status}: {error}")
                        return False
        
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return False
        
        # Rate limiting between messages
        if i < len(messages) - 1:
            await asyncio.sleep(1)
    
    return True


def build_comprehensive_message(
    market_data: Dict[str, Dict],
    stats: Dict,
    sector_performance: Dict,
    top_gainers: List,
    top_losers: List,
    volatility_leaders: List,
    volume_leaders: List
) -> str:
    """Build comprehensive market overview message in selected language(s)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    language = os.getenv("LANGUAGE", "EN").upper()
    
    messages = []
    
    # ENGLISH VERSION
    if language in ["EN", "BOTH"]:
        message_en = f"**📊 Complete S&P 500 Market Overview - {timestamp}**\n\n"
        
        message_en += "**📈 Market Summary**\n"
        message_en += f"• Total Stocks: **{stats['total_stocks']}**\n"
        message_en += f"• Gainers: **{stats['gainers']}** ({stats['gainers']/stats['total_stocks']*100:.1f}%)\n"
        message_en += f"• Losers: **{stats['losers']}** ({stats['losers']/stats['total_stocks']*100:.1f}%)\n"
        message_en += f"• Average Change: **{stats['avg_change']:+.2f}%**\n"
        message_en += f"• Market Sentiment: **{stats['sentiment']}**\n"
        message_en += f"• Volatility (Std Dev): **{stats['std_dev']:.2f}%**\n\n"
        
        message_en += "**🚀 Top 10 Gainers**\n"
        for ticker, change in top_gainers:
            if not pd.isna(change) and ticker in market_data:
                formatted = format_enhanced_ticker(ticker, market_data[ticker])
                message_en += f"• {formatted}\n"
        message_en += "\n"
        
        message_en += "**📉 Top 10 Losers**\n"
        for ticker, change in top_losers:
            if not pd.isna(change) and ticker in market_data:
                formatted = format_enhanced_ticker(ticker, market_data[ticker])
                message_en += f"• {formatted}\n"
        message_en += "\n"
        
        message_en += "**🏢 Sector Performance**\n"
        for sector, data in sector_performance.items():
            emoji = "🟢" if data['avg_change'] > 0 else "🔴"
            message_en += f"{emoji} **{sector}**: {data['avg_change']:+.2f}% "
            message_en += f"(Best: {data['best']:+.2f}%, Worst: {data['worst']:+.2f}%)\n"
        message_en += "\n"
        
        message_en += "**⚡ Highest Volatility (Intraday Range)**\n"
        for ticker, vol, change in volatility_leaders:
            if not pd.isna(vol) and not pd.isna(change) and ticker in market_data:
                company = COMPANY_NAMES.get(ticker, ticker)
                message_en += f"• **{company} | {ticker}**: **{vol:.2f}%** range ({change:+.2f}% change)\n"
        message_en += "\n"
        
        message_en += "**📊 Highest Volume**\n"
        for ticker, volume, change in volume_leaders:
            if not pd.isna(change):
                company = COMPANY_NAMES.get(ticker, ticker)
                vol_millions = volume / 1_000_000
                message_en += f"• **{company} | {ticker}**: **{vol_millions:.1f}M** shares ({change:+.2f}%)\n"
        message_en += "\n"
        
        advance_decline = stats['gainers'] - stats['losers']
        message_en += "**📊 Market Breadth**\n"
        message_en += f"• Advance/Decline: **{advance_decline:+d}**\n"
        
        if abs(advance_decline) > 100:
            if advance_decline > 0:
                message_en += "• **Strong bullish breadth** - Broad market strength\n"
            else:
                message_en += "• **Strong bearish breadth** - Broad market weakness\n"
        else:
            message_en += "• **Mixed breadth** - Selective moves\n"

        # Enhanced analysis
        enhanced_analysis = explain_volatility_and_sentiment(stats)
        message_en += enhanced_analysis
        
        message_en += "\n_Automated by GitHub Actions • All S&P 500 Variations_"
        messages.append(message_en)
    
    # FRENCH VERSION
    if language in ["FR", "BOTH"]:
        message_fr = f"**📊 Vue Complète du Marché S&P 500 - {timestamp}**\n\n"
        
        message_fr += "**📈 Résumé du Marché**\n"
        message_fr += f"• Total Actions: **{stats['total_stocks']}**\n"
        message_fr += f"• Hausses: **{stats['gainers']}** ({stats['gainers']/stats['total_stocks']*100:.1f}%)\n"
        message_fr += f"• Baisses: **{stats['losers']}** ({stats['losers']/stats['total_stocks']*100:.1f}%)\n"
        message_fr += f"• Variation Moyenne: **{stats['avg_change']:+.2f}%**\n"
        message_fr += f"• Sentiment du Marché: **{stats['sentiment']}**\n"
        message_fr += f"• Volatilité (Écart-type): **{stats['std_dev']:.2f}%**\n\n"
        
        message_fr += "**🚀 Top 10 Hausses**\n"
        for ticker, change in top_gainers:
            if not pd.isna(change) and ticker in market_data:
                formatted = format_enhanced_ticker(ticker, market_data[ticker])
                message_en += f"• {formatted}\n"
        message_en += "\n"
        
        message_fr += "**📉 Top 10 Baisses**\n"
        for ticker, change in top_losers:
            if not pd.isna(change) and ticker in market_data:
                formatted = format_enhanced_ticker(ticker, market_data[ticker])
                message_en += f"• {formatted}\n"
        message_en += "\n"
        
        sector_translations = {
            'Technology': 'Technologie',
            'Finance': 'Finance',
            'Healthcare': 'Santé',
            'Consumer': 'Consommation',
            'Energy': 'Énergie',
            'Industrial': 'Industrie'
        }
        
        message_fr += "**🏢 Performance par Secteur**\n"
        for sector, data in sector_performance.items():
            emoji = "🟢" if data['avg_change'] > 0 else "🔴"
            sector_fr = sector_translations.get(sector, sector)
            message_fr += f"{emoji} **{sector_fr}**: {data['avg_change']:+.2f}% "
            message_fr += f"(Meilleur: {data['best']:+.2f}%, Pire: {data['worst']:+.2f}%)\n"
        message_fr += "\n"
        
        message_fr += "**⚡ Plus Haute Volatilité (Amplitude Journalière)**\n"
        for ticker, vol, change in volatility_leaders:
            if not pd.isna(vol) and not pd.isna(change) and ticker in market_data:
                company = COMPANY_NAMES.get(ticker, ticker)
                message_en += f"• **{company} | {ticker}**: **{vol:.2f}%** range ({change:+.2f}% change)\n"
        message_en += "\n"
        
        message_fr += "**📊 Plus Fort Volume**\n"
        for ticker, volume, change in volume_leaders:
            if not pd.isna(change):
                company = COMPANY_NAMES.get(ticker, ticker)
                vol_millions = volume / 1_000_000
                message_en += f"• **{company} | {ticker}**: **{vol_millions:.1f}M** shares ({change:+.2f}%)\n"
        message_en += "\n"
        
        advance_decline = stats['gainers'] - stats['losers']
        message_fr += "**📊 Étendue du Marché**\n"
        message_fr += f"• Hausses/Baisses: **{advance_decline:+d}**\n"
        
        if abs(advance_decline) > 100:
            if advance_decline > 0:
                message_fr += "• **Forte tendance haussière** - Force généralisée du marché\n"
            else:
                message_fr += "• **Forte tendance baissière** - Faiblesse généralisée du marché\n"
        else:
            message_fr += "• **Tendance mixte** - Mouvements sélectifs\n"
        
        message_fr += "\n_Automatisé par GitHub Actions • Toutes Variations S&P 500_"
        messages.append(message_fr)
    
    # Combine with separator if both languages
    if language == "BOTH":
        return "\n\n" + "─" * 50 + "\n\n".join(messages)
    else:
        return messages[0]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("S&P 500 All Variations Monitor - GitHub Actions")
    logger.info("="*60)
    
    try:
        # Validate configuration
        if not DISCORD_WEBHOOK_URL:
            raise ValueError("DISCORD_WEBHOOK_ALL_VARIATIONS environment variable not set")

        # Load company names
        global COMPANY_NAMES
        COMPANY_NAMES = get_company_names()
        
        # Step 1: Get tickers
        tickers = get_sp500_tickers()
        
        # Step 2: Fetch ALL market data
        market_data = await fetch_all_market_data(tickers)
        
        if not market_data:
            logger.warning("No market data available - markets may be closed")
            await send_discord_message(
                "⚠️ **S&P 500 All Variations**: No market data available. Markets may be closed."
            )
            return
        
        # Step 3: Calculate statistics
        stats = calculate_market_statistics(market_data)
        
        # Step 4: Analyze by sector
        sector_performance = analyze_by_sector(market_data)
        
        # Step 5: Get top performers
        top_gainers, top_losers = get_top_performers(market_data, n=10)
        
        # Step 6: Get volatility and volume leaders
        volatility_leaders = get_volatility_leaders(market_data, n=5)
        volume_leaders = get_volume_leaders(market_data, n=5)
        
        # Step 7: Build and send comprehensive message
        message = build_comprehensive_message(
            market_data,
            stats,
            sector_performance,
            top_gainers,
            top_losers,
            volatility_leaders,
            volume_leaders
        )

        # Track prices for next update
        track_price_history(market_data)
        
        await send_discord_message(message)
        
        logger.info("="*60)
        logger.info("✓ Execution completed successfully")
        logger.info(f"  Total stocks analyzed: {len(market_data)}")
        logger.info(f"  Market sentiment: {stats['sentiment']}")
        logger.info(f"  Average change: {stats['avg_change']:+.2f}%")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        
        try:
            await send_discord_message(
                f"⚠️ **S&P 500 All Variations Error**\n```{str(e)}```"
            )
        except:
            pass
        
        raise


if __name__ == "__main__":
    asyncio.run(main())
