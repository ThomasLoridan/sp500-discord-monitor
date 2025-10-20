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
    """Build comprehensive market overview message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    message = f"**📊 Complete S&P 500 Market Overview - {timestamp}**\n\n"
    
    # Market Summary
    message += "**📈 Market Summary**\n"
    message += f"• Total Stocks: **{stats['total_stocks']}**\n"
    message += f"• Gainers: **{stats['gainers']}** ({stats['gainers']/stats['total_stocks']*100:.1f}%)\n"
    message += f"• Losers: **{stats['losers']}** ({stats['losers']/stats['total_stocks']*100:.1f}%)\n"
    message += f"• Average Change: **{stats['avg_change']:+.2f}%**\n"
    message += f"• Market Sentiment: **{stats['sentiment']}**\n"
    message += f"• Volatility (Std Dev): **{stats['std_dev']:.2f}%**\n\n"
    
    # Top 10 Gainers
    message += "**🚀 Top 10 Gainers**\n"
    for ticker, change in top_gainers:
        message += f"• {ticker}: **+{change:.2f}%**\n"
    message += "\n"
    
    # Top 10 Losers
    message += "**📉 Top 10 Losers**\n"
    for ticker, change in top_losers:
        message += f"• {ticker}: **{change:.2f}%**\n"
    message += "\n"
    
    # Sector Performance
    message += "**🏢 Sector Performance**\n"
    for sector, data in sector_performance.items():
        emoji = "🟢" if data['avg_change'] > 0 else "🔴"
        message += f"{emoji} **{sector}**: {data['avg_change']:+.2f}% "
        message += f"(Best: {data['best']:+.2f}%, Worst: {data['worst']:+.2f}%)\n"
    message += "\n"
    
    # Volatility Leaders
    message += "**⚡ Highest Volatility (Intraday Range)**\n"
    for ticker, vol, change in volatility_leaders:
        message += f"• {ticker}: **{vol:.2f}%** range ({change:+.2f}% change)\n"
    message += "\n"
    
    # Volume Leaders
    message += "**📊 Highest Volume**\n"
    for ticker, volume, change in volume_leaders:
        vol_millions = volume / 1_000_000
        message += f"• {ticker}: **{vol_millions:.1f}M** shares ({change:+.2f}%)\n"
    message += "\n"
    
    # Market Breadth Analysis
    advance_decline = stats['gainers'] - stats['losers']
    message += "**📊 Market Breadth**\n"
    message += f"• Advance/Decline: **{advance_decline:+d}**\n"
    
    if abs(advance_decline) > 100:
        if advance_decline > 0:
            message += "• **Strong bullish breadth** - Broad market strength\n"
        else:
            message += "• **Strong bearish breadth** - Broad market weakness\n"
    else:
        message += "• **Mixed breadth** - Selective moves\n"
    
    message += "\n_Automated by GitHub Actions • All S&P 500 Variations_"
    
    return message


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
