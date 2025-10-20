#!/usr/bin/env python3
"""
S&P 500 Discord Monitor big Movers - Single Script for GitHub Actions
Optimized for serverless execution with all dependencies in one file.
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
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TOP_MOVER_PERCENTILE = float(os.getenv("TOP_MOVER_PERCENTILE", "75.0"))
HISTORICAL_DAYS = int(os.getenv("HISTORICAL_DAYS", "5"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        # Fallback: try yfinance
        try:
            logger.info("Trying yfinance fallback...")
            import yfinance as yf
            if hasattr(yf, 'tickers_sp500'):
                return sorted({to_ticker_symbol(s) for s in yf.tickers_sp500()})
        except:
            pass
        
        raise RuntimeError("Failed to fetch S&P 500 tickers")


# ============================================================================
# DATA FETCHING
# ============================================================================

async def fetch_market_data(tickers: List[str], batch_size: int = 50) -> Dict[str, Dict]:
    """
    Fetch current day's market data for all tickers.
    Returns dict with ticker -> {price, change_pct, volume, etc.}
    """
    logger.info(f"Fetching market data for {len(tickers)} tickers...")
    
    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        try:
            # Download batch data
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
                    
                    # Calculate metrics
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
        
        # Small delay between batches
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.5)
    
    logger.info(f"✓ Fetched data for {len(results)} stocks")
    return results


async def fetch_historical_data(tickers: List[str], days: int = 5) -> Dict[str, pd.DataFrame]:
    """Fetch historical daily data for pattern analysis."""
    logger.info(f"Fetching {days} days of historical data...")
    
    results = {}
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        try:
            data = yf.download(
                batch,
                period=f"{days}d",
                interval="1d",
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
                    
                    if not ticker_data.empty and len(ticker_data) >= 2:
                        results[ticker] = ticker_data
                
                except Exception as e:
                    logger.debug(f"Error processing historical {ticker}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Historical batch error: {e}")
            continue
        
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.5)
    
    logger.info(f"✓ Fetched historical data for {len(results)} stocks")
    return results


# ============================================================================
# MARKET ANALYSIS
# ============================================================================

def get_top_movers(market_data: Dict[str, Dict], percentile: float = 75.0) -> Dict:
    """Identify top movers above the specified percentile."""
    if not market_data:
        return {'gainers': [], 'losers': []}
    
    changes = [(ticker, data['change_pct']) for ticker, data in market_data.items()]
    
    if not changes:
        return {'gainers': [], 'losers': []}
    
    changes_array = np.array([c[1] for c in changes])
    
    # Calculate percentile thresholds
    upper_threshold = np.percentile(changes_array, percentile)
    lower_threshold = np.percentile(changes_array, 100 - percentile)
    
    # Filter top gainers and losers
    gainers = [(ticker, change) for ticker, change in changes 
               if change >= upper_threshold and change > 0]
    losers = [(ticker, change) for ticker, change in changes 
              if change <= lower_threshold and change < 0]
    
    # Sort by absolute change
    gainers.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(gainers)} top gainers, {len(losers)} top losers")
    
    return {
        'gainers': gainers,
        'losers': losers,
        'threshold_upper': upper_threshold,
        'threshold_lower': lower_threshold
    }


def analyze_patterns(historical_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze high/low patterns for predictive insights.
    
    Rules:
    1. If Friday high < Thursday high → Friday low likely revisited Monday
    2. If Wednesday high < Monday high → Wednesday low likely revisited Thursday
    """
    patterns = {}
    today = datetime.now().weekday()  # 0=Monday, 4=Friday
    
    # Pattern 1: Friday-Thursday (run on Friday or later)
    if today >= 4:
        friday_thursday = analyze_friday_thursday_pattern(historical_data)
        if friday_thursday:
            patterns['friday_thursday_pattern'] = friday_thursday
    
    # Pattern 2: Wednesday-Monday (run on Wednesday or later)
    if today >= 2:
        wednesday_monday = analyze_wednesday_monday_pattern(historical_data)
        if wednesday_monday:
            patterns['wednesday_monday_pattern'] = wednesday_monday
    
    return patterns


def analyze_friday_thursday_pattern(historical_data: Dict[str, pd.DataFrame]) -> Dict | None:
    """If Friday high < Thursday high, predict Friday low revisit on Monday."""
    candidates = []
    
    for ticker, df in historical_data.items():
        if len(df) < 5:
            continue
        
        try:
            recent = df.tail(5)
            
            if len(recent) < 2:
                continue
            
            friday_data = recent.iloc[-1]
            thursday_data = recent.iloc[-2]
            
            friday_high = friday_data['High']
            thursday_high = thursday_data['High']
            friday_low = friday_data['Low']
            
            if friday_high < thursday_high:
                high_diff_pct = ((thursday_high - friday_high) / thursday_high) * 100
                
                candidates.append({
                    'ticker': ticker,
                    'friday_high': float(friday_high),
                    'thursday_high': float(thursday_high),
                    'friday_low': float(friday_low),
                    'high_diff_pct': float(high_diff_pct),
                    'predicted_level': float(friday_low)
                })
        
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x['high_diff_pct'], reverse=True)
        return candidates[0]
    
    return None


def analyze_wednesday_monday_pattern(historical_data: Dict[str, pd.DataFrame]) -> Dict | None:
    """If Wednesday high < Monday high, predict Wednesday low revisit on Thursday."""
    candidates = []
    
    for ticker, df in historical_data.items():
        if len(df) < 5:
            continue
        
        try:
            recent = df.tail(5)
            
            if len(recent) < 3:
                continue
            
            monday_data = recent.iloc[0]
            wednesday_data = recent.iloc[2] if len(recent) > 2 else None
            
            if wednesday_data is None:
                continue
            
            monday_high = monday_data['High']
            wednesday_high = wednesday_data['High']
            wednesday_low = wednesday_data['Low']
            
            if wednesday_high < monday_high:
                high_diff_pct = ((monday_high - wednesday_high) / monday_high) * 100
                
                candidates.append({
                    'ticker': ticker,
                    'monday_high': float(monday_high),
                    'wednesday_high': float(wednesday_high),
                    'wednesday_low': float(wednesday_low),
                    'high_diff_pct': float(high_diff_pct),
                    'predicted_level': float(wednesday_low)
                })
        
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            continue
    
    if candidates:
        candidates.sort(key=lambda x: x['high_diff_pct'], reverse=True)
        return candidates[0]
    
    return None


# ============================================================================
# DISCORD NOTIFICATION
# ============================================================================

async def send_discord_message(content: str) -> bool:
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.error("DISCORD_WEBHOOK_URL not set")
        return False
    
    # Discord 2000 character limit
    if len(content) > 2000:
        content = content[:1997] + "..."
    
    payload = {
        "content": content,
        "username": "SP500 Monitor",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2920/2920349.png"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DISCORD_WEBHOOK_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 204:
                    logger.info("✓ Discord message sent successfully")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Discord error {response.status}: {error}")
                    return False
    
    except Exception as e:
        logger.error(f"Error sending Discord message: {e}")
        return False


def build_message(top_movers: Dict, patterns: Dict) -> str:
    """Build formatted Discord message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    message = f"**📊 S&P 500 Market Update - {timestamp}**\n\n"
    
    # Top Gainers
    if top_movers.get('gainers'):
        message += "**🚀 Top Gainers (>75th percentile)**\n"
        for ticker, change in top_movers['gainers'][:5]:
            message += f"• {ticker}: **+{change:.2f}%**\n"
        message += "\n"
    
    # Top Losers
    if top_movers.get('losers'):
        message += "**📉 Top Losers (>75th percentile)**\n"
        for ticker, change in top_movers['losers'][:5]:
            message += f"• {ticker}: **{change:.2f}%**\n"
        message += "\n"
    
    # Market Pattern Analysis
    if patterns:
        message += "**🔍 Market Pattern Analysis**\n"
        
        if patterns.get('friday_thursday_pattern'):
            p = patterns['friday_thursday_pattern']
            message += f"⚠️ Friday high < Thursday high detected\n"
            message += f"• Likely revisit Friday low: **${p['friday_low']:.2f}** on Monday\n"
            message += f"• Key ticker: **{p['ticker']}**\n\n"
        
        if patterns.get('wednesday_monday_pattern'):
            p = patterns['wednesday_monday_pattern']
            message += f"⚠️ Wednesday high < Monday high detected\n"
            message += f"• Likely revisit Wednesday low: **${p['wednesday_low']:.2f}** on Thursday\n"
            message += f"• Key ticker: **{p['ticker']}**\n\n"
    
    message += f"_Automated by GitHub Actions_"
    return message


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("S&P 500 Discord Monitor - GitHub Actions")
    logger.info("="*60)
    
    try:
        # Validate configuration
        if not DISCORD_WEBHOOK_URL:
            raise ValueError("DISCORD_WEBHOOK_URL environment variable not set")
        
        # Step 1: Get tickers
        tickers = get_sp500_tickers()
        
        # Step 2: Fetch current market data
        market_data = await fetch_market_data(tickers)
        
        if not market_data:
            logger.warning("No market data available - markets may be closed")
            await send_discord_message(
                "⚠️ **S&P 500 Monitor**: No market data available. Markets may be closed."
            )
            return
        
        # Step 3: Analyze top movers
        top_movers = get_top_movers(market_data, percentile=TOP_MOVER_PERCENTILE)
        
        # Step 4: Fetch historical data for pattern analysis
        # Use only top movers + random sample for efficiency
        analysis_tickers = []
        if top_movers['gainers']:
            analysis_tickers.extend([t for t, _ in top_movers['gainers'][:10]])
        if top_movers['losers']:
            analysis_tickers.extend([t for t, _ in top_movers['losers'][:10]])
        
        # Add random sample
        import random
        additional = random.sample(tickers, min(30, len(tickers)))
        analysis_tickers.extend(additional)
        analysis_tickers = list(set(analysis_tickers))  # Remove duplicates
        
        historical_data = await fetch_historical_data(analysis_tickers, days=HISTORICAL_DAYS)
        
        # Step 5: Pattern analysis
        patterns = analyze_patterns(historical_data)
        
        # Step 6: Build and send message
        message = build_message(top_movers, patterns)
        await send_discord_message(message)
        
        logger.info("="*60)
        logger.info("✓ Execution completed successfully")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        
        # Try to send error notification
        try:
            await send_discord_message(
                f"⚠️ **S&P 500 Monitor Error**\n```{str(e)}```"
            )
        except:
            pass
        
        raise


if __name__ == "__main__":
    asyncio.run(main())
