#!/usr/bin/env python3
"""
S&P 500 Fundamental Analysis Scanner
Runs at market open (9:30 AM) and close (4:00 PM) ET
Calculates P/E ratio, MktCap/Earnings ratio, and investment recommendations
"""

import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
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
DISCORD_WEBHOOK_FUNDAMENTAL = os.getenv("DISCORD_WEBHOOK_FUNDAMENTAL")
LANGUAGE = os.getenv("LANGUAGE", "EN").upper()
TOP_N_OPPORTUNITIES = int(os.getenv("TOP_N_OPPORTUNITIES", "20"))

TIMEZONE_US_EASTERN = ZoneInfo("America/New_York")
TIMEZONE_FRANCE = ZoneInfo("Europe/Paris")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA FETCHING
# ============================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers."""
    logger.info("Fetching S&P 500 tickers...")
    try:
        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns:
                tickers = sorted({s.replace(".", "-").upper().strip() for s in tbl["Symbol"].astype(str)})
                logger.info(f"✓ Loaded {len(tickers)} tickers")
                return tickers
    except Exception as e:
        logger.error(f"Error: {e}")
    return []


def get_company_names_and_sectors() -> Dict[str, Dict]:
    """Fetch company names and sectors."""
    try:
        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), flavor="lxml")
        for tbl in tables:
            if "Symbol" in tbl.columns and "Security" in tbl.columns:
                mapping = {}
                for _, row in tbl.iterrows():
                    ticker = str(row["Symbol"]).replace(".", "-").upper().strip()
                    company = str(row["Security"]).strip()
                    sector = str(row.get("GICS Sector", "Unknown")).strip()
                    mapping[ticker] = {
                        'name': company,
                        'sector': sector
                    }
                logger.info(f"✓ Loaded {len(mapping)} company details")
                return mapping
    except Exception as e:
        logger.error(f"Error: {e}")
    return {}


async def fetch_fundamental_data(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch fundamental data for all tickers.
    Returns: {ticker: {price, market_cap, earnings, pe_ratio, etc.}}
    """
    logger.info(f"Fetching fundamental data for {len(tickers)} tickers...")
    
    results = {}
    batch_size = 20  # Smaller batches for fundamental data
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract key metrics
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                market_cap = info.get('marketCap')
                eps = info.get('trailingEps')  # Earnings per share (trailing 12 months)
                pe_ratio = info.get('trailingPE')
                forward_pe = info.get('forwardPE')
                earnings = info.get('netIncomeToCommon')  # Net earnings
                revenue = info.get('totalRevenue')
                profit_margin = info.get('profitMargins')
                
                # Skip if critical data missing
                if not price or not market_cap:
                    continue
                
                # Calculate MktCap/Earnings ratio if earnings available
                if earnings and earnings > 0:
                    mktcap_earnings_ratio = (market_cap / earnings) * 100
                else:
                    mktcap_earnings_ratio = None
                
                # Calculate P/E if not available
                if not pe_ratio and eps and eps > 0:
                    pe_ratio = price / eps
                
                results[ticker] = {
                    'price': price,
                    'market_cap': market_cap,
                    'earnings_per_share': eps,
                    'pe_ratio': pe_ratio,
                    'forward_pe': forward_pe,
                    'net_earnings': earnings,
                    'mktcap_earnings_ratio': mktcap_earnings_ratio,
                    'revenue': revenue,
                    'profit_margin': profit_margin,
                    'has_fundamental_data': True
                }
            
            except Exception as e:
                logger.debug(f"Error fetching {ticker}: {e}")
                continue
        
        # Rate limiting
        await asyncio.sleep(0.5)
    
    logger.info(f"✓ Fetched fundamental data for {len(results)} stocks")
    return results


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_investment_score(data: Dict) -> float:
    """
    Calculate investment score (0-100).
    Lower P/E and MktCap/Earnings = Better score
    """
    score = 50  # Base score
    
    pe_ratio = data.get('pe_ratio')
    mktcap_earnings = data.get('mktcap_earnings_ratio')
    profit_margin = data.get('profit_margin')
    
    # P/E scoring (lower is better)
    if pe_ratio:
        if pe_ratio < 10:
            score += 20
        elif pe_ratio < 15:
            score += 15
        elif pe_ratio < 20:
            score += 10
        elif pe_ratio < 25:
            score += 5
        elif pe_ratio > 50:
            score -= 20
        elif pe_ratio > 40:
            score -= 10
    
    # MktCap/Earnings scoring (lower is better)
    if mktcap_earnings:
        if mktcap_earnings < 50:
            score += 15
        elif mktcap_earnings < 100:
            score += 10
        elif mktcap_earnings < 200:
            score += 5
        elif mktcap_earnings > 500:
            score -= 15
    
    # Profit margin bonus
    if profit_margin:
        if profit_margin > 0.20:  # 20%+
            score += 10
        elif profit_margin > 0.15:
            score += 5
        elif profit_margin < 0.05:
            score -= 10
    
    # Ensure score is within 0-100
    return max(0, min(100, score))


def generate_investment_comment(data: Dict, score: float) -> str:
    """Generate investment comment based on ratios."""
    pe_ratio = data.get('pe_ratio')
    mktcap_earnings = data.get('mktcap_earnings_ratio')
    profit_margin = data.get('profit_margin', 0)
    
    comments = []
    
    # P/E analysis
    if pe_ratio:
        if pe_ratio < 10:
            comments.append("🟢 Very low P/E - potentially undervalued")
        elif pe_ratio < 15:
            comments.append("🟢 Low P/E - good value")
        elif pe_ratio < 25:
            comments.append("🟡 Moderate P/E - fairly valued")
        elif pe_ratio < 40:
            comments.append("🟠 High P/E - growth premium")
        else:
            comments.append("🔴 Very high P/E - potentially overvalued")
    
    # MktCap/Earnings analysis
    if mktcap_earnings:
        if mktcap_earnings < 100:
            comments.append("✅ Strong earnings relative to valuation")
        elif mktcap_earnings < 200:
            comments.append("➖ Moderate earnings coverage")
        else:
            comments.append("⚠️ High valuation relative to earnings")
    
    # Profitability
    if profit_margin > 0.15:
        comments.append("💰 High profit margins")
    elif profit_margin > 0.08:
        comments.append("💵 Healthy profit margins")
    elif profit_margin > 0:
        comments.append("📊 Low profit margins")
    else:
        comments.append("📉 Negative margins - unprofitable")
    
    return " | ".join(comments) if comments else "Insufficient data"


def generate_investment_indicator(score: float) -> str:
    """Generate investment indicator emoji and text."""
    if score >= 80:
        return "🟢 STRONG BUY"
    elif score >= 70:
        return "🟢 BUY"
    elif score >= 60:
        return "🟡 HOLD/BUY"
    elif score >= 50:
        return "🟡 HOLD"
    elif score >= 40:
        return "🟠 HOLD/AVOID"
    else:
        return "🔴 AVOID"


def rank_investment_opportunities(fundamental_data: Dict, company_info: Dict) -> List[Dict]:
    """Rank stocks by investment potential."""
    opportunities = []
    
    for ticker, data in fundamental_data.items():
        if not data.get('pe_ratio') and not data.get('mktcap_earnings_ratio'):
            continue
        
        score = calculate_investment_score(data)
        comment = generate_investment_comment(data, score)
        indicator = generate_investment_indicator(score)
        
        info = company_info.get(ticker, {})
        
        opportunities.append({
            'ticker': ticker,
            'company': info.get('name', ticker),
            'sector': info.get('sector', 'Unknown'),
            'price': data['price'],
            'market_cap': data['market_cap'],
            'pe_ratio': data.get('pe_ratio'),
            'mktcap_earnings_ratio': data.get('mktcap_earnings_ratio'),
            'profit_margin': data.get('profit_margin'),
            'score': score,
            'comment': comment,
            'indicator': indicator
        })
    
    # Sort by score (highest first)
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    return opportunities


# ============================================================================
# MESSAGE BUILDING
# ============================================================================

def build_fundamental_analysis_message(
    opportunities: List[Dict],
    stats: Dict,
    session: str  # "MARKET OPEN" or "MARKET CLOSE"
) -> str:
    """Build comprehensive fundamental analysis message."""
    
    now_eastern = datetime.now(TIMEZONE_US_EASTERN)
    now_france = datetime.now(TIMEZONE_FRANCE)
    
    msg = f"**💼 S&P 500 FUNDAMENTAL ANALYSIS**\n"
    msg += f"**{session} - {now_eastern.strftime('%A, %B %d, %Y')}**\n"
    msg += f"⏰ {now_eastern.strftime('%I:%M %p %Z')} | {now_france.strftime('%H:%M %Z')}\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**📊 ANALYSIS OVERVIEW**\n\n"
    msg += f"• Total Stocks Analyzed: **{stats['total_analyzed']}**\n"
    msg += f"• Stocks with Complete Data: **{stats['complete_data']}**\n"
    msg += f"• Average P/E Ratio: **{stats['avg_pe']:.1f}**\n"
    msg += f"• Median P/E Ratio: **{stats['median_pe']:.1f}**\n"
    msg += f"• Strong Buy Opportunities: **{stats['strong_buy_count']}**\n"
    msg += f"• Buy Opportunities: **{stats['buy_count']}**\n\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"**🎯 TOP {TOP_N_OPPORTUNITIES} INVESTMENT OPPORTUNITIES**\n"
    msg += "_Ranked by fundamental strength (low P/E & MktCap/Earnings)_\n\n"
    
    # Build table
    msg += "```\n"
    msg += f"{'Rank':<5} {'Ticker':<6} {'P/E':<7} {'M/E%':<7} {'Score':<6} {'Signal':<12}\n"
    msg += "="*50 + "\n"
    
    for i, opp in enumerate(opportunities[:TOP_N_OPPORTUNITIES], 1):
        pe_str = f"{opp['pe_ratio']:.1f}" if opp['pe_ratio'] else "N/A"
        me_str = f"{opp['mktcap_earnings_ratio']:.0f}" if opp['mktcap_earnings_ratio'] else "N/A"
        
        signal = opp['indicator'].split()[1] if len(opp['indicator'].split()) > 1 else "HOLD"
        
        msg += f"{i:<5} {opp['ticker']:<6} {pe_str:<7} {me_str:<7} {opp['score']:<6.0f} {signal:<12}\n"
    
    msg += "```\n\n"
    
    # Detailed top 10
    msg += "**📋 DETAILED ANALYSIS - TOP 10**\n\n"
    
    for i, opp in enumerate(opportunities[:10], 1):
        msg += f"**{i}. {opp['company']} ({opp['ticker']})**\n"
        msg += f"   • Price: **${opp['price']:.2f}**\n"
        msg += f"   • Market Cap: **${opp['market_cap']/1e9:.2f}B**\n"
        msg += f"   • Sector: **{opp['sector']}**\n"
        
        if opp['pe_ratio']:
            msg += f"   • P/E Ratio: **{opp['pe_ratio']:.2f}**\n"
        
        if opp['mktcap_earnings_ratio']:
            msg += f"   • MktCap/Earnings: **{opp['mktcap_earnings_ratio']:.1f}%**\n"
        
        if opp['profit_margin']:
            msg += f"   • Profit Margin: **{opp['profit_margin']*100:.1f}%**\n"
        
        msg += f"   • Investment Score: **{opp['score']:.0f}/100**\n"
        msg += f"   • Signal: **{opp['indicator']}**\n"
        msg += f"   • Analysis: _{opp['comment']}_\n\n"
    
    # Value vs Growth breakdown
    value_stocks = [o for o in opportunities if o.get('pe_ratio') and o['pe_ratio'] < 15]
    growth_stocks = [o for o in opportunities if o.get('pe_ratio') and o['pe_ratio'] > 30]
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**📊 MARKET CLASSIFICATION**\n\n"
    msg += f"**Value Stocks** (P/E < 15): **{len(value_stocks)}**\n"
    if value_stocks[:5]:
        msg += "Top 5: " + ", ".join([v['ticker'] for v in value_stocks[:5]]) + "\n\n"
    
    msg += f"**Growth Stocks** (P/E > 30): **{len(growth_stocks)}**\n"
    if growth_stocks[:5]:
        msg += "Top 5: " + ", ".join([g['ticker'] for g in growth_stocks[:5]]) + "\n\n"
    
    # Investment strategy
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += "**💡 INVESTMENT STRATEGY GUIDE**\n\n"
    
    msg += "**Key Metrics Explained:**\n"
    msg += "• **P/E Ratio**: Price/Earnings - Lower = Better value\n"
    msg += "  - Under 15: Value territory 🟢\n"
    msg += "  - 15-25: Fair value 🟡\n"
    msg += "  - Over 30: Growth premium 🔴\n\n"
    
    msg += "• **MktCap/Earnings %**: Market Cap / Net Earnings × 100\n"
    msg += "  - Under 100: Strong earnings 🟢\n"
    msg += "  - 100-200: Moderate 🟡\n"
    msg += "  - Over 200: Expensive 🔴\n\n"
    
    msg += "**Investment Signals:**\n"
    msg += "• 🟢 **STRONG BUY** (80+): Excellent fundamentals\n"
    msg += "• 🟢 **BUY** (70-79): Good fundamentals\n"
    msg += "• 🟡 **HOLD** (50-69): Fair value, monitor\n"
    msg += "• 🔴 **AVOID** (<50): Poor fundamentals\n\n"
    
    msg += "**⚠️ IMPORTANT DISCLAIMERS:**\n"
    msg += "• Fundamental analysis is ONE tool - not financial advice\n"
    msg += "• Low P/E can indicate problems, not just value\n"
    msg += "• Consider technical analysis, news, and market conditions\n"
    msg += "• Always do your own research before investing\n"
    msg += "• Past performance doesn't guarantee future results\n\n"
    
    if session == "MARKET OPEN":
        msg += "**🌅 Morning Analysis Complete**\n"
        msg += "• Review opportunities before market action\n"
        msg += "• Set alerts on high-score stocks\n"
        msg += "• Monitor price action at open\n"
    else:
        msg += "**🌆 End-of-Day Analysis Complete**\n"
        msg += "• Review day's performance vs fundamentals\n"
        msg += "• Plan tomorrow's watchlist\n"
        msg += "• Evaluate entry/exit points overnight\n"
    
    msg += "\n_📊 Fundamental Analysis • Updated " + session + "_"
    
    return msg


def calculate_stats(opportunities: List[Dict]) -> Dict:
    """Calculate summary statistics."""
    pe_ratios = [o['pe_ratio'] for o in opportunities if o.get('pe_ratio')]
    
    strong_buy = sum(1 for o in opportunities if o['score'] >= 80)
    buy = sum(1 for o in opportunities if 70 <= o['score'] < 80)
    complete_data = sum(1 for o in opportunities if o.get('pe_ratio') and o.get('mktcap_earnings_ratio'))
    
    return {
        'total_analyzed': len(opportunities),
        'complete_data': complete_data,
        'avg_pe': np.mean(pe_ratios) if pe_ratios else 0,
        'median_pe': np.median(pe_ratios) if pe_ratios else 0,
        'strong_buy_count': strong_buy,
        'buy_count': buy
    }


async def send_discord_message(content: str) -> bool:
    """Send message to Discord."""
    if not DISCORD_WEBHOOK_FUNDAMENTAL:
        logger.error("DISCORD_WEBHOOK_FUNDAMENTAL not set")
        return False
    
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    
    for chunk in chunks:
        payload = {
            "content": chunk,
            "username": "S&P 500 Fundamental Analysis",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2920/2920349.png"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DISCORD_WEBHOOK_FUNDAMENTAL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 204:
                        return False
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    logger.info("✓ Fundamental analysis sent")
    return True


async def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("S&P 500 Fundamental Analysis")
    logger.info("="*60)
    
    try:
        if not DISCORD_WEBHOOK_FUNDAMENTAL:
            raise ValueError("DISCORD_WEBHOOK_FUNDAMENTAL not set")
        
        # Determine session
        now = datetime.now(TIMEZONE_US_EASTERN)
        if 9 <= now.hour < 12:
            session = "MARKET OPEN"
        else:
            session = "MARKET CLOSE"
        
        logger.info(f"Running {session} analysis...")
        
        # Get data
        company_info = get_company_names_and_sectors()
        tickers = get_sp500_tickers()
        
        # Fetch fundamental data
        fundamental_data = await fetch_fundamental_data(tickers)
        
        if not fundamental_data:
            logger.warning("No fundamental data available")
            return
        
        # Rank opportunities
        opportunities = rank_investment_opportunities(fundamental_data, company_info)
        
        if not opportunities:
            logger.warning("No opportunities found")
            return
        
        # Calculate stats
        stats = calculate_stats(opportunities)
        
        # Build and send message
        message = build_fundamental_analysis_message(opportunities, stats, session)
        
        await send_discord_message(message)
        
        logger.info(f"✓ {session} fundamental analysis completed")
        logger.info(f"  Analyzed: {len(opportunities)} stocks")
        logger.info(f"  Strong Buy: {stats['strong_buy_count']}")
        logger.info(f"  Buy: {stats['buy_count']}")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
