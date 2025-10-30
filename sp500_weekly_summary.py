#!/usr/bin/env python3
"""
S&P 500 Weekly Market Summary
Runs every Saturday to analyze past week's market evolution
Comprehensive volatility, sector performance, and trend analysis
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict
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
DISCORD_WEBHOOK_WEEKLY = os.getenv("DISCORD_WEBHOOK_WEEKLY")
LANGUAGE = os.getenv("LANGUAGE", "EN").upper()

TIMEZONE_US_EASTERN = ZoneInfo("America/New_York")
TIMEZONE_FRANCE = ZoneInfo("Europe/Paris")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sector classifications (same as daily)
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG'],
    'Consumer Staples': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],
    'Industrials': ['BA', 'HON', 'UNP', 'CAT', 'RTX', 'GE', 'LMT', 'DE', 'MMM', 'UPS'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'ALB', 'NUE'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'DLR', 'O', 'WELL', 'AVB'],
    'Communication Services': ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED']
}


def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers."""
    try:
        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns:
                return sorted({s.replace(".", "-").upper().strip() for s in tbl["Symbol"].astype(str)})
    except Exception as e:
        logger.error(f"Error: {e}")
    return []


def get_company_names() -> Dict[str, str]:
    """Fetch company names."""
    try:
        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns and "Security" in tbl.columns:
                return {
                    str(row["Symbol"]).replace(".", "-").upper().strip(): str(row["Security"]).strip()
                    for _, row in tbl.iterrows()
                }
    except Exception as e:
        logger.error(f"Error: {e}")
    return {}


async def fetch_weekly_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch past week's trading data."""
    logger.info(f"Fetching weekly data for {len(tickers)} tickers...")
    
    results = {}
    batch_size = 100
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                period="1mo",
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
                    
                    if not ticker_data.empty and len(ticker_data) >= 5:
                        results[ticker] = ticker_data.tail(10)  # Last 2 weeks for context
                
                except Exception as e:
                    continue
        except Exception as e:
            logger.error(f"Batch error: {e}")
            continue
    
    logger.info(f"✓ Fetched data for {len(results)} stocks")
    return results


def analyze_weekly_performance(weekly_data: Dict) -> Dict:
    """Analyze week's overall performance."""
    week_changes = []
    volatilities = []
    volumes = []
    
    for ticker, df in weekly_data.items():
        if len(df) < 5:
            continue
        
        try:
            # Last 5 trading days = 1 week
            week_df = df.tail(5)
            
            week_start = week_df.iloc[0]['Close']
            week_end = week_df.iloc[-1]['Close']
            week_change = ((week_end - week_start) / week_start) * 100
            week_changes.append(week_change)
            
            # Volatility (std dev of daily returns)
            daily_returns = week_df['Close'].pct_change().dropna() * 100
            volatility = daily_returns.std()
            volatilities.append(volatility)
            
            # Average volume
            avg_volume = week_df['Volume'].mean()
            volumes.append(avg_volume)
        
        except Exception as e:
            continue
    
    if not week_changes:
        return {}
    
    gainers = sum(1 for c in week_changes if c > 0)
    losers = sum(1 for c in week_changes if c < 0)
    
    return {
        'total_stocks': len(week_changes),
        'gainers': gainers,
        'losers': losers,
        'avg_change': np.mean(week_changes),
        'median_change': np.median(week_changes),
        'best_performer': max(week_changes),
        'worst_performer': min(week_changes),
        'avg_volatility': np.mean(volatilities),
        'median_volatility': np.median(volatilities),
        'high_volatility_stocks': sum(1 for v in volatilities if v > 3.0),
        'total_volume': sum(volumes),
        'avg_volume': np.mean(volumes)
    }


def analyze_sector_weekly(weekly_data: Dict, company_names: Dict) -> Dict:
    """Analyze sector performance for the week."""
    sector_data = defaultdict(lambda: {'changes': [], 'volatilities': [], 'tickers': []})
    
    for ticker, df in weekly_data.items():
        if len(df) < 5:
            continue
        
        for sector, sector_tickers in SECTORS.items():
            if ticker in sector_tickers:
                try:
                    week_df = df.tail(5)
                    week_change = ((week_df.iloc[-1]['Close'] - week_df.iloc[0]['Close']) / 
                                   week_df.iloc[0]['Close']) * 100
                    
                    daily_returns = week_df['Close'].pct_change().dropna() * 100
                    volatility = daily_returns.std()
                    
                    sector_data[sector]['changes'].append(week_change)
                    sector_data[sector]['volatilities'].append(volatility)
                    sector_data[sector]['tickers'].append((ticker, week_change))
                
                except Exception as e:
                    continue
                break
    
    # Calculate sector metrics
    sector_summary = {}
    for sector, data in sector_data.items():
        if data['changes']:
            sector_summary[sector] = {
                'avg_change': np.mean(data['changes']),
                'median_change': np.median(data['changes']),
                'avg_volatility': np.mean(data['volatilities']),
                'stock_count': len(data['changes']),
                'gainers': sum(1 for c in data['changes'] if c > 0),
                'losers': sum(1 for c in data['changes'] if c < 0),
                'best': max(data['tickers'], key=lambda x: x[1]),
                'worst': min(data['tickers'], key=lambda x: x[1])
            }
    
    return dict(sorted(sector_summary.items(), key=lambda x: x[1]['avg_change'], reverse=True))


def identify_weekly_leaders_laggards(weekly_data: Dict, company_names: Dict) -> Dict:
    """Identify week's top performers and laggards."""
    performances = []
    
    for ticker, df in weekly_data.items():
        if len(df) < 5:
            continue
        
        try:
            week_df = df.tail(5)
            week_change = ((week_df.iloc[-1]['Close'] - week_df.iloc[0]['Close']) / 
                           week_df.iloc[0]['Close']) * 100
            
            performances.append({
                'ticker': ticker,
                'company': company_names.get(ticker, ticker),
                'change': week_change,
                'close': week_df.iloc[-1]['Close']
            })
        
        except Exception as e:
            continue
    
    performances.sort(key=lambda x: x['change'], reverse=True)
    
    return {
        'leaders': performances[:10],
        'laggards': performances[-10:]
    }


def analyze_multi_timeframe_trends(weekly_data: Dict) -> Dict:
    """Analyze trends across week, month, quarter."""
    trends = {
        'week': [],
        'month': [],
        'quarter': []
    }
    
    for ticker, df in weekly_data.items():
        try:
            # Week (5 days)
            if len(df) >= 5:
                week_change = ((df.iloc[-1]['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close']) * 100
                trends['week'].append(week_change)
            
            # Month (20 days)
            if len(df) >= 20:
                month_change = ((df.iloc[-1]['Close'] - df.iloc[-20]['Close']) / df.iloc[-20]['Close']) * 100
                trends['month'].append(month_change)
        
        except Exception as e:
            continue
    
    result = {}
    for period, changes in trends.items():
        if changes:
            result[period] = {
                'avg_change': np.mean(changes),
                'median_change': np.median(changes),
                'up_stocks': sum(1 for c in changes if c > 0),
                'down_stocks': sum(1 for c in changes if c < 0),
                'total': len(changes)
            }
    
    return result


def build_weekly_summary_message(
    weekly_stats: Dict,
    sector_performance: Dict,
    leaders_laggards: Dict,
    trends: Dict,
    company_names: Dict
) -> str:
    """Build comprehensive weekly summary message."""
    
    now_eastern = datetime.now(TIMEZONE_US_EASTERN)
    now_france = datetime.now(TIMEZONE_FRANCE)
    
    # Get week date range
    week_end = now_eastern - timedelta(days=now_eastern.weekday() + 1)  # Last Friday
    week_start = week_end - timedelta(days=4)  # Previous Monday
    
    msg = f"**📊 S&P 500 WEEKLY MARKET SUMMARY**\n"
    msg += f"**Week of {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}**\n"
    msg += f"⏰ {now_eastern.strftime('%I:%M %p %Z')} | {now_france.strftime('%H:%M %Z')}\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**📈 WEEK IN REVIEW**\n\n"
    
    # Overall performance
    avg_change = weekly_stats['avg_change']
    emoji = "🟢" if avg_change > 0 else "🔴"
    
    msg += f"**Market Performance**: {emoji} **{avg_change:+.2f}%**\n"
    msg += f"• Best Performer: **+{weekly_stats['best_performer']:.2f}%**\n"
    msg += f"• Worst Performer: **{weekly_stats['worst_performer']:.2f}%**\n"
    msg += f"• Median Change: **{weekly_stats['median_change']:+.2f}%**\n\n"
    
    msg += f"**Market Breadth**:\n"
    msg += f"• Winners: **{weekly_stats['gainers']}** ({weekly_stats['gainers']/weekly_stats['total_stocks']*100:.1f}%)\n"
    msg += f"• Losers: **{weekly_stats['losers']}** ({weekly_stats['losers']/weekly_stats['total_stocks']*100:.1f}%)\n"
    msg += f"• W/L Ratio: **{weekly_stats['gainers']/weekly_stats['losers']:.2f}**\n\n"
    
    msg += f"**Volatility**:\n"
    msg += f"• Average: **{weekly_stats['avg_volatility']:.2f}%**\n"
    msg += f"• High Volatility Stocks (>3%): **{weekly_stats['high_volatility_stocks']}**\n"
    
    if weekly_stats['avg_volatility'] > 2.5:
        msg += f"• ⚠️ **High volatility week** - Increased risk\n"
    elif weekly_stats['avg_volatility'] < 1.5:
        msg += f"• ✅ **Low volatility week** - Stable conditions\n"
    msg += "\n"
    
    msg += f"**Volume**:\n"
    msg += f"• Total: **{weekly_stats['total_volume']/1e9:.2f}B** shares\n"
    msg += f"• Daily Average: **{weekly_stats['avg_volume']/1e6:.1f}M** per stock\n\n"
    
    # Sector Performance
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**🏢 SECTOR PERFORMANCE**\n\n"
    
    for i, (sector, data) in enumerate(list(sector_performance.items())[:5], 1):
        emoji = "🟢" if data['avg_change'] > 0 else "🔴"
        msg += f"**{i}. {sector}** {emoji}\n"
        msg += f"   • Performance: **{data['avg_change']:+.2f}%**\n"
        msg += f"   • Volatility: **{data['avg_volatility']:.2f}%**\n"
        msg += f"   • Winners/Losers: **{data['gainers']}/{data['losers']}**\n"
        
        best_ticker, best_change = data['best']
        worst_ticker, worst_change = data['worst']
        msg += f"   • Best: {company_names.get(best_ticker, best_ticker)} ({best_change:+.2f}%)\n"
        msg += f"   • Worst: {company_names.get(worst_ticker, worst_ticker)} ({worst_change:+.2f}%)\n\n"
    
    # Bottom 3 sectors
    msg += "**Weakest Sectors**:\n"
    for i, (sector, data) in enumerate(list(sector_performance.items())[-3:], 1):
        msg += f"{i}. **{sector}**: {data['avg_change']:+.2f}%\n"
    msg += "\n"
    
    # Leaders and Laggards
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**🚀 WEEK'S TOP 10 LEADERS**\n"
    for i, stock in enumerate(leaders_laggards['leaders'], 1):
        msg += f"{i}. **{stock['company']}** ({stock['ticker']}): **+{stock['change']:.2f}%**\n"
    msg += "\n"
    
    msg += "**📉 WEEK'S TOP 10 LAGGARDS**\n"
    for i, stock in enumerate(leaders_laggards['laggards'], 1):
        msg += f"{i}. **{stock['company']}** ({stock['ticker']}): **{stock['change']:.2f}%**\n"
    msg += "\n"
    
    # Multi-timeframe trends
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**📊 MULTI-TIMEFRAME ANALYSIS**\n\n"
    
    for period, data in trends.items():
        period_name = period.capitalize()
        emoji = "🟢" if data['avg_change'] > 0 else "🔴"
        
        msg += f"**{period_name}** {emoji}\n"
        msg += f"   • Average: **{data['avg_change']:+.2f}%**\n"
        msg += f"   • Up/Down: **{data['up_stocks']}/{data['down_stocks']}**\n"
        
        if period == 'week':
            if abs(data['avg_change']) > 3:
                strength = "Strong"
            elif abs(data['avg_change']) > 1.5:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "Bullish" if data['avg_change'] > 0 else "Bearish"
            msg += f"   • Trend: **{strength} {direction}**\n"
        
        msg += "\n"
    
    # Market outlook
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**🔮 WEEK AHEAD OUTLOOK**\n\n"
    
    week_trend = trends.get('week', {})
    if week_trend:
        avg = week_trend['avg_change']
        
        if avg > 2:
            msg += "• **Bullish momentum** - Strong weekly gains\n"
            msg += "• Watch for continuation or profit-taking\n"
            msg += "• Consider trailing stops to protect gains\n"
        elif avg > 1:
            msg += "• **Positive trend** - Moderate weekly gains\n"
            msg += "• Look for pullback entry opportunities\n"
            msg += "• Momentum may continue into next week\n"
        elif avg > -1:
            msg += "• **Consolidation phase** - Mixed signals\n"
            msg += "• Wait for clearer direction\n"
            msg += "• Good time for watchlist building\n"
        elif avg > -2:
            msg += "• **Bearish pressure** - Moderate weekly losses\n"
            msg += "• Look for oversold bounce candidates\n"
            msg += "• Defensive positioning recommended\n"
        else:
            msg += "• **Strong selling** - Significant weekly losses\n"
            msg += "• Exercise caution with new positions\n"
            msg += "• Wait for stabilization signals\n"
    
    msg += "\n"
    
    # Volatility insight
    if weekly_stats['avg_volatility'] > 2.5:
        msg += "• **High volatility** - Use wider stops, smaller positions\n"
    
    # Breadth insight
    winners_pct = (weekly_stats['gainers'] / weekly_stats['total_stocks']) * 100
    if winners_pct > 60:
        msg += "• **Strong breadth** - Broad market participation\n"
    elif winners_pct < 40:
        msg += "• **Weak breadth** - Narrow market, selective plays only\n"
    
    msg += "\n**💡 Key Levels to Watch**:\n"
    msg += "• Monitor sector rotation patterns\n"
    msg += "• Watch volume on Monday for trend confirmation\n"
    msg += "• Friday-Monday pattern may be active (see daily recaps)\n"
    
    msg += "\n\n_📈 Weekly Summary • Have a great weekend!_"
    
    return msg


async def send_discord_message(content: str) -> bool:
    """Send message to Discord."""
    if not DISCORD_WEBHOOK_WEEKLY:
        logger.error("DISCORD_WEBHOOK_WEEKLY not set")
        return False
    
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    
    for chunk in chunks:
        payload = {
            "content": chunk,
            "username": "S&P 500 Weekly Summary",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2920/2920349.png"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DISCORD_WEBHOOK_WEEKLY,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 204:
                        return False
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    logger.info("✓ Weekly summary sent")
    return True


async def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("S&P 500 Weekly Market Summary")
    logger.info("="*60)
    
    try:
        if not DISCORD_WEBHOOK_WEEKLY:
            raise ValueError("DISCORD_WEBHOOK_WEEKLY not set")
        
        company_names = get_company_names()
        tickers = get_sp500_tickers()
        
        weekly_data = await fetch_weekly_data(tickers)
        
        if not weekly_data:
            logger.warning("No data available")
            return
        
        weekly_stats = analyze_weekly_performance(weekly_data)
        sector_performance = analyze_sector_weekly(weekly_data, company_names)
        leaders_laggards = identify_weekly_leaders_laggards(weekly_data, company_names)
        trends = analyze_multi_timeframe_trends(weekly_data)
        
        message = build_weekly_summary_message(
            weekly_stats,
            sector_performance,
            leaders_laggards,
            trends,
            company_names
        )
        
        await send_discord_message(message)
        
        logger.info("✓ Weekly summary completed")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
