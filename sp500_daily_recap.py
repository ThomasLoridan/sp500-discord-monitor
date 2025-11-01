#!/usr/bin/env python3
"""
S&P 500 Daily Market Close Recap
Runs 10 minutes after market close (4:10 PM ET)
Provides comprehensive day analysis with volatility, sectors, and key drivers
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import aiohttp
from io import StringIO
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DISCORD_WEBHOOK_DAILY_RECAP = os.getenv("DISCORD_WEBHOOK_DAILY_RECAP")
LANGUAGE = os.getenv("LANGUAGE", "EN").upper()

# Timezones
TIMEZONE_UTC = ZoneInfo("UTC")
TIMEZONE_US_EASTERN = ZoneInfo("America/New_York")
TIMEZONE_FRANCE = ZoneInfo("Europe/Paris")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sector classifications
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 
                   'CRM', 'AVGO', 'QCOM', 'TXN', 'ORCL', 'NFLX', 'PYPL', 'AMAT', 'LRCX', 'KLAC'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 
                   'PNC', 'TFC', 'COF', 'BK', 'STT', 'AIG', 'MET', 'PRU', 'ALL', 'TRV'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY', 
                   'AMGN', 'CVS', 'CI', 'ELV', 'GILD', 'REGN', 'ISRG', 'VRTX', 'ZTS', 'SYK'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
                               'CMG', 'ORLY', 'MAR', 'GM', 'F', 'DHI', 'LEN', 'YUM', 'ROST', 'AZO'],
    'Consumer Staples': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
                        'GIS', 'K', 'HSY', 'SYY', 'KHC', 'STZ', 'TAP', 'CPB', 'CAG', 'CHD'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
               'KMI', 'WMB', 'HAL', 'BKR', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA', 'OVV'],
    'Industrials': ['BA', 'HON', 'UNP', 'CAT', 'RTX', 'GE', 'LMT', 'DE', 'MMM', 'UPS',
                    'GD', 'NOC', 'ETN', 'TDG', 'FDX', 'NSC', 'CSX', 'WM', 'EMR', 'PCAR'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'ALB', 'NUE',
                  'VMC', 'MLM', 'CF', 'MOS', 'PPG', 'IFF', 'BALL', 'AVY', 'PKG', 'IP'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'DLR', 'O', 'WELL', 'AVB',
                    'EQR', 'SBAC', 'VTR', 'ARE', 'INVH', 'MAA', 'ESS', 'SUI', 'UDR', 'HST'],
    'Communication Services': ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
                               'EA', 'TTWO', 'NWSA', 'NWS', 'DISH', 'PARA', 'LYV', 'FOXA', 'FOX', 'OMC'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
                  'WEC', 'ES', 'ETR', 'AWK', 'DTE', 'PPL', 'FE', 'AEE', 'CMS', 'CNP']
}


# ============================================================================
# DATA FETCHING
# ============================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    logger.info("Fetching S&P 500 tickers...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns:
                tickers = sorted({s.replace(".", "-").upper().strip() for s in tbl["Symbol"].astype(str)})
                logger.info(f"✓ Loaded {len(tickers)} tickers")
                return tickers
        raise RuntimeError("Couldn't parse tickers")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def get_company_names() -> Dict[str, str]:
    """Fetch company names from Wikipedia."""
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
        logger.error(f"Error: {e}")
        return {}


async def fetch_daily_data(tickers: List[str]) -> Dict[str, Dict]:
    """Fetch full day's trading data with robust validation."""
    logger.info(f"Fetching daily data for {len(tickers)} tickers...")
    
    results = {}
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
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
                    
                    # Extract values with validation
                    open_price = ticker_data['Open'].iloc[0]
                    close_price = ticker_data['Close'].iloc[-1]
                    high = ticker_data['High'].max()
                    low = ticker_data['Low'].min()
                    volume = ticker_data['Volume'].sum()
                    
                    # Strict validation - skip if ANY value is invalid
                    if pd.isna(open_price) or pd.isna(close_price) or pd.isna(high) or pd.isna(low):
                        logger.debug(f"Skipping {ticker}: NaN values detected")
                        continue
                    
                    if open_price <= 0 or close_price <= 0:
                        logger.debug(f"Skipping {ticker}: Zero or negative prices")
                        continue
                    
                    # Calculate metrics
                    daily_change = ((close_price - open_price) / open_price) * 100
                    intraday_range = ((high - low) / open_price) * 100
                    
                    # Validate calculated values
                    if pd.isna(daily_change) or pd.isna(intraday_range):
                        logger.debug(f"Skipping {ticker}: Calculated NaN")
                        continue
                    
                    if np.isinf(daily_change) or np.isinf(intraday_range):
                        logger.debug(f"Skipping {ticker}: Infinite values")
                        continue
                    
                    results[ticker] = {
                        'open': float(open_price),
                        'close': float(close_price),
                        'high': float(high),
                        'low': float(low),
                        'volume': int(volume),
                        'daily_change': float(daily_change),
                        'intraday_range': float(intraday_range)
                    }
                
                except Exception as e:
                    logger.debug(f"Error processing {ticker}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            continue
        
        await asyncio.sleep(0.5)
    
    logger.info(f"✓ Fetched valid data for {len(results)} stocks")
    return results



async def fetch_weekly_historical(tickers: List[str], days: int = 7) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for trend analysis."""
    logger.info(f"Fetching {days} days of historical data...")
    
    results = {}
    batch_size = 100
    
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
                    continue
        
        except Exception as e:
            logger.error(f"Error fetching historical batch: {e}")
            continue
    
    logger.info(f"✓ Fetched historical data for {len(results)} stocks")
    return results


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_day_performance(daily_data: Dict) -> Dict:
    """Comprehensive daily performance analysis with NaN protection."""
    if not daily_data:
        return {
            'total_stocks': 0,
            'gainers': 0,
            'losers': 0,
            'unchanged': 0,
            'avg_change': 0.0,
            'median_change': 0.0,
            'std_dev': 0.0,
            'total_volume': 0,
            'avg_volume': 0.0,
            'avg_volatility': 0.0,
            'sentiment': '⚪ No Data',
            'advance_decline': 0,
            'advance_decline_ratio': 0.0
        }
    
    # Filter out any remaining NaN values
    valid_changes = [d['daily_change'] for d in daily_data.values() 
                     if not pd.isna(d['daily_change']) and not np.isinf(d['daily_change'])]
    valid_volumes = [d['volume'] for d in daily_data.values() 
                     if not pd.isna(d['volume']) and d['volume'] > 0]
    valid_volatilities = [d['intraday_range'] for d in daily_data.values() 
                          if not pd.isna(d['intraday_range']) and not np.isinf(d['intraday_range'])]
    
    if not valid_changes:
        return {
            'total_stocks': len(daily_data),
            'gainers': 0,
            'losers': 0,
            'unchanged': 0,
            'avg_change': 0.0,
            'median_change': 0.0,
            'std_dev': 0.0,
            'total_volume': sum(valid_volumes) if valid_volumes else 0,
            'avg_volume': np.mean(valid_volumes) if valid_volumes else 0.0,
            'avg_volatility': 0.0,
            'sentiment': '⚪ Insufficient Data',
            'advance_decline': 0,
            'advance_decline_ratio': 0.0
        }
    
    gainers = sum(1 for c in valid_changes if c > 0)
    losers = sum(1 for c in valid_changes if c < 0)
    unchanged = len(valid_changes) - gainers - losers
    
    avg_change = float(np.mean(valid_changes))
    median_change = float(np.median(valid_changes))
    std_dev = float(np.std(valid_changes))
    
    total_volume = int(sum(valid_volumes)) if valid_volumes else 0
    avg_volume = float(np.mean(valid_volumes)) if valid_volumes else 0.0
    
    avg_volatility = float(np.mean(valid_volatilities)) if valid_volatilities else 0.0
    
    # Market sentiment
    if avg_change > 1.0:
        sentiment = "🟢 STRONGLY BULLISH"
    elif avg_change > 0.3:
        sentiment = "🟢 Bullish"
    elif avg_change > -0.3:
        sentiment = "⚪ Neutral"
    elif avg_change > -1.0:
        sentiment = "🔴 Bearish"
    else:
        sentiment = "🔴 STRONGLY BEARISH"
    
    advance_decline = gainers - losers
    advance_decline_ratio = gainers / losers if losers > 0 else float('inf')
    
    return {
        'total_stocks': len(daily_data),
        'gainers': gainers,
        'losers': losers,
        'unchanged': unchanged,
        'avg_change': avg_change,
        'median_change': median_change,
        'std_dev': std_dev,
        'total_volume': total_volume,
        'avg_volume': avg_volume,
        'avg_volatility': avg_volatility,
        'sentiment': sentiment,
        'advance_decline': advance_decline,
        'advance_decline_ratio': advance_decline_ratio
    }


def analyze_sector_performance(daily_data: Dict, company_names: Dict) -> Dict:
    """Analyze performance by sector."""
    sector_data = defaultdict(lambda: {'changes': [], 'volumes': [], 'tickers': []})
    
    # Classify stocks by sector
    for ticker, data in daily_data.items():
        for sector, sector_tickers in SECTORS.items():
            if ticker in sector_tickers:
                sector_data[sector]['changes'].append(data['daily_change'])
                sector_data[sector]['volumes'].append(data['volume'])
                sector_data[sector]['tickers'].append(ticker)
                break
    
    # Calculate sector metrics
    sector_summary = {}
    for sector, data in sector_data.items():
        if data['changes']:
            sector_summary[sector] = {
                'avg_change': np.mean(data['changes']),
                'median_change': np.median(data['changes']),
                'total_volume': sum(data['volumes']),
                'stock_count': len(data['changes']),
                'gainers': sum(1 for c in data['changes'] if c > 0),
                'losers': sum(1 for c in data['changes'] if c < 0),
                'best_performer': max(zip(data['tickers'], data['changes']), key=lambda x: x[1]),
                'worst_performer': min(zip(data['tickers'], data['changes']), key=lambda x: x[1])
            }
    
    # Sort by performance
    sorted_sectors = sorted(sector_summary.items(), key=lambda x: x[1]['avg_change'], reverse=True)
    
    return dict(sorted_sectors)


def identify_key_drivers(daily_data: Dict, company_names: Dict, top_n: int = 10) -> Dict:
    """Identify key market drivers (biggest movers with high volume)."""
    # Score = |daily_change| * log(volume)
    scored_stocks = []
    
    for ticker, data in daily_data.items():
        if data['volume'] > 0:
            score = abs(data['daily_change']) * np.log10(data['volume'])
            scored_stocks.append({
                'ticker': ticker,
                'company': company_names.get(ticker, ticker),
                'change': data['daily_change'],
                'volume': data['volume'],
                'score': score
            })
    
    # Sort by score
    scored_stocks.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'top_drivers': scored_stocks[:top_n],
        'biggest_gainers': sorted(
            [s for s in scored_stocks if s['change'] > 0],
            key=lambda x: x['change'],
            reverse=True
        )[:5],
        'biggest_losers': sorted(
            [s for s in scored_stocks if s['change'] < 0],
            key=lambda x: x['change']
        )[:5]
    }


def analyze_friday_monday_pattern(historical_data: Dict) -> Dict:
    """
    Analyze Friday high → Monday low pattern.
    If Friday high < Thursday high → Monday likely revisits Friday low
    """
    patterns_detected = []
    
    for ticker, df in historical_data.items():
        if len(df) < 5:
            continue
        
        try:
            df['weekday'] = pd.to_datetime(df.index).dayofweek
            
            # Get most recent Friday (4) and Thursday (3)
            friday_rows = df[df['weekday'] == 4]
            thursday_rows = df[df['weekday'] == 3]
            
            if len(friday_rows) == 0 or len(thursday_rows) == 0:
                continue
            
            friday = friday_rows.iloc[-1]
            thursday = thursday_rows.iloc[-1]
            
            # Check pattern
            if friday['High'] < thursday['High']:
                high_diff_pct = ((thursday['High'] - friday['High']) / thursday['High']) * 100
                potential_drop = ((friday['Close'] - friday['Low']) / friday['Close']) * 100
                
                if high_diff_pct > 0.5:  # At least 0.5% difference
                    patterns_detected.append({
                        'ticker': ticker,
                        'friday_high': friday['High'],
                        'thursday_high': thursday['High'],
                        'friday_low': friday['Low'],
                        'friday_close': friday['Close'],
                        'pattern_strength': high_diff_pct,
                        'potential_drop': potential_drop,
                        'prediction': 'Monday may revisit Friday low'
                    })
        
        except Exception as e:
            continue
    
    # Sort by pattern strength
    patterns_detected.sort(key=lambda x: x['pattern_strength'], reverse=True)
    
    return {
        'pattern_detected': len(patterns_detected) > 0,
        'total_stocks': len(patterns_detected),
        'top_candidates': patterns_detected[:10]
    }


def calculate_trend_analysis(historical_data: Dict) -> Dict:
    """Calculate week, month, quarter trends."""
    trends = {
        'week': {'up': 0, 'down': 0, 'avg_change': 0},
        'month': {'up': 0, 'down': 0, 'avg_change': 0},
        'quarter': {'up': 0, 'down': 0, 'avg_change': 0}
    }
    
    week_changes = []
    
    for ticker, df in historical_data.items():
        if len(df) < 2:
            continue
        
        try:
            # Week trend (last 5 trading days)
            if len(df) >= 5:
                week_start = df.iloc[-5]['Close']
                week_end = df.iloc[-1]['Close']
                week_change = ((week_end - week_start) / week_start) * 100
                week_changes.append(week_change)
                
                if week_change > 0:
                    trends['week']['up'] += 1
                else:
                    trends['week']['down'] += 1
        
        except Exception as e:
            continue
    
    if week_changes:
        trends['week']['avg_change'] = np.mean(week_changes)
    
    return trends


# ============================================================================
# MESSAGE BUILDING
# ============================================================================

def build_daily_recap_message(
    daily_stats: Dict,
    sector_performance: Dict,
    key_drivers: Dict,
    friday_pattern: Dict,
    trends: Dict,
    company_names: Dict
) -> str:
    """Build comprehensive daily recap message."""
    
    now_eastern = datetime.now(TIMEZONE_US_EASTERN)
    now_france = datetime.now(TIMEZONE_FRANCE)
    
    language = LANGUAGE
    
    if language in ["EN", "BOTH"]:
        msg = f"**📊 S&P 500 DAILY MARKET RECAP**\n"
        msg += f"**{now_eastern.strftime('%A, %B %d, %Y')} - Market Close**\n"
        msg += f"⏰ {now_eastern.strftime('%I:%M %p %Z')} | {now_france.strftime('%H:%M %Z')}\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**📈 MARKET OVERVIEW**\n"
        msg += f"• Market Sentiment: **{daily_stats['sentiment']}**\n"
        msg += f"• Average Change: **{daily_stats['avg_change']:+.2f}%**\n"
        msg += f"• Median Change: **{daily_stats['median_change']:+.2f}%**\n"
        msg += f"• Volatility (Std Dev): **{daily_stats['std_dev']:.2f}%**\n"
        msg += f"• Average Intraday Range: **{daily_stats['avg_volatility']:.2f}%**\n\n"
        
        msg += "**📊 MARKET BREADTH**\n"
        msg += f"• Gainers: **{daily_stats['gainers']}** ({daily_stats['gainers']/daily_stats['total_stocks']*100:.1f}%)\n"
        msg += f"• Losers: **{daily_stats['losers']}** ({daily_stats['losers']/daily_stats['total_stocks']*100:.1f}%)\n"
        msg += f"• Unchanged: **{daily_stats['unchanged']}**\n"
        msg += f"• Advance/Decline: **{daily_stats['advance_decline']:+d}**\n"
        msg += f"• A/D Ratio: **{daily_stats['advance_decline_ratio']:.2f}**\n\n"
        
        msg += f"**💼 VOLUME**\n"
        msg += f"• Total Volume: **{daily_stats['total_volume']/1e9:.2f}B** shares\n"
        msg += f"• Average Per Stock: **{daily_stats['avg_volume']/1e6:.1f}M** shares\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**🏢 SECTOR PERFORMANCE (Top 5)**\n"
        for i, (sector, data) in enumerate(list(sector_performance.items())[:5], 1):
            emoji = "🟢" if data['avg_change'] > 0 else "🔴"
            msg += f"{i}. {emoji} **{sector}**: {data['avg_change']:+.2f}%\n"
            msg += f"   • Stocks: {data['gainers']}/{data['stock_count']} up\n"
            best_ticker, best_change = data['best_performer']
            msg += f"   • Best: {company_names.get(best_ticker, best_ticker)} ({best_change:+.2f}%)\n"
        msg += "\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**🎯 KEY MARKET DRIVERS (Top 5)**\n"
        for i, driver in enumerate(key_drivers['top_drivers'][:5], 1):
            emoji = "🟢" if driver['change'] > 0 else "🔴"
            msg += f"{i}. {emoji} **{driver['company']} ({driver['ticker']})**\n"
            msg += f"   • Change: **{driver['change']:+.2f}%**\n"
            msg += f"   • Volume: **{driver['volume']/1e6:.1f}M** shares\n"
        msg += "\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**🚀 BIGGEST GAINERS**\n"
        for i, stock in enumerate(key_drivers['biggest_gainers'], 1):
            msg += f"{i}. **{stock['company']}** ({stock['ticker']}): **+{stock['change']:.2f}%**\n"
        msg += "\n"
        
        msg += "**📉 BIGGEST LOSERS**\n"
        for i, stock in enumerate(key_drivers['biggest_losers'], 1):
            msg += f"{i}. **{stock['company']}** ({stock['ticker']}): **{stock['change']:.2f}%**\n"
        msg += "\n"
        
        # Friday-Monday Pattern
        if friday_pattern['pattern_detected']:
            msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            msg += "**⚠️ FRIDAY-MONDAY PATTERN DETECTED**\n"
            msg += f"• Stocks Showing Pattern: **{friday_pattern['total_stocks']}**\n"
            msg += "• Prediction: **Monday may see weakness in these stocks**\n\n"
            msg += "**Top 5 Candidates for Monday:**\n"
            for i, pattern in enumerate(friday_pattern['top_candidates'][:5], 1):
                msg += f"{i}. **{company_names.get(pattern['ticker'], pattern['ticker'])}**\n"
                msg += f"   • Pattern Strength: {pattern['pattern_strength']:.1f}%\n"
                msg += f"   • Potential Drop: {pattern['potential_drop']:.1f}%\n"
                msg += f"   • Target: ${pattern['friday_low']:.2f}\n"
            msg += "\n"
        
        # Trend Analysis
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**📊 TREND ANALYSIS**\n"
        msg += f"• **Past Week**: {trends['week']['avg_change']:+.2f}% avg | "
        msg += f"{trends['week']['up']} up, {trends['week']['down']} down\n"
        
        trend_emoji = "🟢" if trends['week']['avg_change'] > 0 else "🔴"
        msg += f"• **Market Direction**: {trend_emoji} "
        if abs(trends['week']['avg_change']) > 2:
            msg += "Strong " + ("Uptrend" if trends['week']['avg_change'] > 0 else "Downtrend")
        elif abs(trends['week']['avg_change']) > 1:
            msg += "Moderate " + ("Uptrend" if trends['week']['avg_change'] > 0 else "Downtrend")
        else:
            msg += "Sideways/Consolidation"
        msg += "\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "**💡 TRADING INSIGHTS**\n"
        
        if daily_stats['avg_change'] > 1.0:
            msg += "• Strong bullish momentum - Consider taking profits\n"
            msg += "• Watch for potential pullback tomorrow\n"
        elif daily_stats['avg_change'] > 0.5:
            msg += "• Positive day - Momentum may continue\n"
            msg += "• Look for continuation patterns\n"
        elif daily_stats['avg_change'] > -0.5:
            msg += "• Mixed signals - Wait for clearer direction\n"
            msg += "• Good day for selective plays\n"
        elif daily_stats['avg_change'] > -1.0:
            msg += "• Bearish pressure - Consider defensive positioning\n"
            msg += "• Look for oversold bounce candidates\n"
        else:
            msg += "• Strong selling pressure - Exercise caution\n"
            msg += "• Wait for stabilization before entering\n"
        
        if daily_stats['avg_volatility'] > 3.0:
            msg += "• High volatility - Wide stops recommended\n"
        
        if abs(daily_stats['advance_decline']) > 200:
            msg += f"• {'Strong breadth supports move' if daily_stats['advance_decline'] > 0 else 'Weak breadth - be cautious'}\n"
        
        msg += "\n_Automated Daily Recap • See you tomorrow! 📈_"
        
        return msg
    
    return ""


# ============================================================================
# DISCORD
# ============================================================================

async def send_discord_message(content: str) -> bool:
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK_DAILY_RECAP:
        logger.error("DISCORD_WEBHOOK_DAILY_RECAP not set")
        return False
    
    # Split if too long
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    
    for chunk in chunks:
        payload = {
            "content": chunk,
            "username": "S&P 500 Daily Recap",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2920/2920349.png"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DISCORD_WEBHOOK_DAILY_RECAP,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 204:
                        logger.error(f"Discord error {response.status}")
                        return False
                    await asyncio.sleep(1)  # Rate limit
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    logger.info("✓ Discord message sent")
    return True


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("S&P 500 Daily Market Recap")
    logger.info("="*60)
    
    try:
        if not DISCORD_WEBHOOK_DAILY_RECAP:
            raise ValueError("DISCORD_WEBHOOK_DAILY_RECAP not set")
        
        # Get data
        company_names = get_company_names()
        tickers = get_sp500_tickers()
        
        # Fetch today's data
        daily_data = await fetch_daily_data(tickers)
        
        if not daily_data:
            logger.warning("No market data available")
            return
        
        # Fetch historical for pattern analysis
        historical_data = await fetch_weekly_historical(tickers, days=7)
        
        # Analyze
        daily_stats = analyze_day_performance(daily_data)
        sector_performance = analyze_sector_performance(daily_data, company_names)
        key_drivers = identify_key_drivers(daily_data, company_names)
        friday_pattern = analyze_friday_monday_pattern(historical_data)
        trends = calculate_trend_analysis(historical_data)
        
        # Build and send message
        message = build_daily_recap_message(
            daily_stats,
            sector_performance,
            key_drivers,
            friday_pattern,
            trends,
            company_names
        )
        
        await send_discord_message(message)
        
        logger.info("✓ Daily recap completed")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
