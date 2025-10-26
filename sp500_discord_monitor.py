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

# ============ ADD THESE NEW LINES HERE ============
COMPANY_NAMES = {}  # Will store ticker -> company name mapping
PREVIOUS_PRICES = {}  # Prices from last update (30 min ago)
OPEN_PRICES = {}  # Prices from market open
YESTERDAY_CLOSES = {}  # Prices from yesterday close
# ==================================================

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
        # Fallback: try yfinance
        try:
            logger.info("Trying yfinance fallback...")
            import yfinance as yf
            if hasattr(yf, 'tickers_sp500'):
                return sorted({to_ticker_symbol(s) for s in yf.tickers_sp500()})
        except:
            pass
        
        raise RuntimeError("Failed to fetch S&P 500 tickers")

def format_enhanced_ticker(ticker: str, data: Dict) -> str:
    """
    Format ticker with company name and multi-timeframe changes.
    
    Returns: "Company Name | TICKER: +X.XX% (+Y.YY% since last | +Z.ZZ% since yesterday)"
    """
    company = COMPANY_NAMES.get(ticker, ticker)
    current = data['current_price']
    
    # Change since market open
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


# ============================================================================
# ENHANCED PATTERN ANALYSIS MODULE
# Add this to sp500_discord_monitor.py
# ============================================================================

from collections import defaultdict
from datetime import datetime, timedelta

def analyze_enhanced_patterns(historical_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Enhanced pattern analysis with company tracking and detailed predictions.
    
    Returns comprehensive pattern data including:
    - Pattern type and detection date
    - Predicted price levels to watch
    - Companies following the pattern
    - Confidence scores
    """
    patterns = {
        'friday_thursday': None,
        'wednesday_monday': None,
        'active_patterns': [],
        'companies_to_watch': []
    }
    
    today = datetime.now()
    current_weekday = today.weekday()  # 0=Monday, 4=Friday
    
    # Pattern 1: Friday-Thursday Analysis
    if current_weekday >= 4 or current_weekday == 0:  # Friday, Saturday, Sunday, or Monday
        friday_thursday = analyze_friday_thursday_detailed(historical_data, today)
        if friday_thursday:
            patterns['friday_thursday'] = friday_thursday
            patterns['active_patterns'].append('friday_thursday')
    
    # Pattern 2: Wednesday-Monday Analysis
    if current_weekday >= 2 or current_weekday == 3:  # Wednesday, Thursday
        wednesday_monday = analyze_wednesday_monday_detailed(historical_data, today)
        if wednesday_monday:
            patterns['wednesday_monday'] = wednesday_monday
            patterns['active_patterns'].append('wednesday_monday')
    
    # Compile companies to watch across all patterns
    all_companies = set()
    if patterns['friday_thursday']:
        all_companies.update([c['ticker'] for c in patterns['friday_thursday']['companies']])
    if patterns['wednesday_monday']:
        all_companies.update([c['ticker'] for c in patterns['wednesday_monday']['companies']])
    
    patterns['companies_to_watch'] = sorted(list(all_companies))
    
    return patterns


def analyze_friday_thursday_detailed(
    historical_data: Dict[str, pd.DataFrame], 
    current_date: datetime
) -> Dict | None:
    """
    Detailed Friday-Thursday pattern analysis.
    
    Pattern: If Friday High < Thursday High → Friday Low will be revisited Monday
    
    Returns:
    - Pattern detected: Yes/No
    - Detection date: When pattern was identified
    - Predicted revisit date: When to watch for the move
    - Target level: The Friday low to watch
    - Companies: List of stocks showing this pattern with details
    - Confidence: Based on historical data and magnitude
    """
    companies_with_pattern = []
    
    for ticker, df in historical_data.items():
        if len(df) < 5:
            continue
        
        try:
            recent = df.tail(5).copy()
            recent['weekday'] = pd.to_datetime(recent.index).dayofweek
            
            # Identify Friday and Thursday
            friday_rows = recent[recent['weekday'] == 4]
            thursday_rows = recent[recent['weekday'] == 3]
            
            if len(friday_rows) == 0 or len(thursday_rows) == 0:
                continue
            
            # Get most recent Friday and Thursday
            friday_data = friday_rows.iloc[-1]
            thursday_data = thursday_rows.iloc[-1]
            
            friday_high = friday_data['High']
            thursday_high = thursday_data['High']
            friday_low = friday_data['Low']
            friday_close = friday_data['Close']
            
            # Check pattern condition
            if friday_high < thursday_high:
                high_diff_pct = ((thursday_high - friday_high) / thursday_high) * 100
                
                # Calculate additional metrics
                friday_range = ((friday_high - friday_low) / friday_low) * 100
                volume_ratio = friday_data.get('Volume', 0) / thursday_data.get('Volume', 1)
                
                # Confidence score (0-100)
                confidence = calculate_pattern_confidence(high_diff_pct, volume_ratio, friday_range)
                
                # Potential downside from current price
                if pd.notna(friday_close) and friday_close > 0:
                    potential_move_pct = ((friday_close - friday_low) / friday_close) * 100
                else:
                    potential_move_pct = 0
                
                companies_with_pattern.append({
                    'ticker': ticker,
                    'friday_high': float(friday_high),
                    'thursday_high': float(thursday_high),
                    'friday_low': float(friday_low),
                    'friday_close': float(friday_close),
                    'target_level': float(friday_low),
                    'high_diff_pct': float(high_diff_pct),
                    'friday_range_pct': float(friday_range),
                    'volume_ratio': float(volume_ratio),
                    'confidence': float(confidence),
                    'potential_move_pct': float(potential_move_pct),
                    'friday_date': friday_data.name.strftime('%Y-%m-%d'),
                    'thursday_date': thursday_data.name.strftime('%Y-%m-%d')
                })
        
        except Exception as e:
            logger.debug(f"Error in Friday-Thursday analysis for {ticker}: {e}")
            continue
    
    if not companies_with_pattern:
        return None
    
    # Sort by confidence score
    companies_with_pattern.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate next Monday
    days_until_monday = (7 - current_date.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = current_date + timedelta(days=days_until_monday)
    
    # Get most recent Friday date from companies
    detection_date = companies_with_pattern[0]['friday_date']
    
    return {
        'pattern_type': 'Friday-Thursday',
        'description': 'Friday High < Thursday High',
        'detected_on': detection_date,
        'predicted_action_date': next_monday.strftime('%Y-%m-%d (Monday)'),
        'prediction': 'Friday low levels will likely be revisited',
        'companies': companies_with_pattern[:10],  # Top 10 by confidence
        'total_companies': len(companies_with_pattern),
        'avg_confidence': sum(c['confidence'] for c in companies_with_pattern) / len(companies_with_pattern)
    }


def analyze_wednesday_monday_detailed(
    historical_data: Dict[str, pd.DataFrame],
    current_date: datetime
) -> Dict | None:
    """
    Detailed Wednesday-Monday pattern analysis.
    
    Pattern: If Wednesday High < Monday High → Wednesday Low will be revisited Thursday
    """
    companies_with_pattern = []
    
    for ticker, df in historical_data.items():
        if len(df) < 5:
            continue
        
        try:
            recent = df.tail(5).copy()
            recent['weekday'] = pd.to_datetime(recent.index).dayofweek
            
            # Identify Wednesday (2) and Monday (0)
            wednesday_rows = recent[recent['weekday'] == 2]
            monday_rows = recent[recent['weekday'] == 0]
            
            if len(wednesday_rows) == 0 or len(monday_rows) == 0:
                continue
            
            wednesday_data = wednesday_rows.iloc[-1]
            monday_data = monday_rows.iloc[-1]
            
            wednesday_high = wednesday_data['High']
            monday_high = monday_data['High']
            wednesday_low = wednesday_data['Low']
            wednesday_close = wednesday_data['Close']
            
            # Check pattern condition
            if wednesday_high < monday_high:
                high_diff_pct = ((monday_high - wednesday_high) / monday_high) * 100
                
                wednesday_range = ((wednesday_high - wednesday_low) / wednesday_low) * 100
                volume_ratio = wednesday_data.get('Volume', 0) / monday_data.get('Volume', 1)
                
                confidence = calculate_pattern_confidence(high_diff_pct, volume_ratio, wednesday_range)
                
                if pd.notna(wednesday_close) and wednesday_close > 0:
                    potential_move_pct = ((wednesday_close - wednesday_low) / wednesday_close) * 100
                else:
                    potential_move_pct = 0
                
                companies_with_pattern.append({
                    'ticker': ticker,
                    'wednesday_high': float(wednesday_high),
                    'monday_high': float(monday_high),
                    'wednesday_low': float(wednesday_low),
                    'wednesday_close': float(wednesday_close),
                    'target_level': float(wednesday_low),
                    'high_diff_pct': float(high_diff_pct),
                    'wednesday_range_pct': float(wednesday_range),
                    'volume_ratio': float(volume_ratio),
                    'confidence': float(confidence),
                    'potential_move_pct': float(potential_move_pct),
                    'wednesday_date': wednesday_data.name.strftime('%Y-%m-%d'),
                    'monday_date': monday_data.name.strftime('%Y-%m-%d')
                })
        
        except Exception as e:
            logger.debug(f"Error in Wednesday-Monday analysis for {ticker}: {e}")
            continue
    
    if not companies_with_pattern:
        return None
    
    companies_with_pattern.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate next Thursday
    current_weekday = current_date.weekday()
    if current_weekday < 3:  # Before Thursday
        days_until_thursday = 3 - current_weekday
    else:  # Thursday or after
        days_until_thursday = (3 - current_weekday) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
    
    next_thursday = current_date + timedelta(days=days_until_thursday)
    
    detection_date = companies_with_pattern[0]['wednesday_date']
    
    return {
        'pattern_type': 'Wednesday-Monday',
        'description': 'Wednesday High < Monday High',
        'detected_on': detection_date,
        'predicted_action_date': next_thursday.strftime('%Y-%m-%d (Thursday)'),
        'prediction': 'Wednesday low levels will likely be revisited',
        'companies': companies_with_pattern[:10],
        'total_companies': len(companies_with_pattern),
        'avg_confidence': sum(c['confidence'] for c in companies_with_pattern) / len(companies_with_pattern)
    }


def calculate_pattern_confidence(
    high_diff_pct: float,
    volume_ratio: float,
    daily_range_pct: float
) -> float:
    """
    Calculate confidence score for pattern prediction (0-100).
    
    Factors:
    - Larger high difference = higher confidence
    - Higher volume = higher confidence
    - Larger daily range = more volatility = moderate confidence impact
    """
    base_confidence = 65  # Historical baseline (~70-75% accuracy)
    
    # High difference contribution (up to +20 points)
    high_diff_bonus = min(high_diff_pct * 3, 20)
    
    # Volume contribution (up to +10 points)
    if volume_ratio > 1.2:  # 20% more volume
        volume_bonus = min((volume_ratio - 1) * 15, 10)
    elif volume_ratio < 0.8:  # 20% less volume
        volume_bonus = -5
    else:
        volume_bonus = 0
    
    # Range contribution (up to +5 points)
    if daily_range_pct > 3:  # Significant intraday movement
        range_bonus = min(daily_range_pct * 0.5, 5)
    else:
        range_bonus = 0
    
    total_confidence = base_confidence + high_diff_bonus + volume_bonus + range_bonus
    
    return min(max(total_confidence, 0), 95)  # Cap between 0-95%


def build_enhanced_pattern_message(patterns: Dict, language: str = "EN") -> str:
    """
    Build detailed pattern analysis section for Discord message.
    """
    if not patterns['active_patterns']:
        return ""
    
    if language == "FR":
        return build_pattern_message_french(patterns)
    else:
        return build_pattern_message_english(patterns)


def build_pattern_message_english(patterns: Dict) -> str:
    """Build English pattern message."""
    message = "**🔍 MARKET SITUATIONAL ANALYSIS**\n"
    message += "_Pattern-Based Trading Signals for Decision Making_\n\n"
    
    # Friday-Thursday Pattern
    if patterns.get('friday_thursday'):
        p = patterns['friday_thursday']
        message += "**📅 Pattern #1: Friday-Thursday Analysis**\n"
        message += f"🎯 **Pattern Detected:** {p['description']}\n"
        message += f"📆 **Detected On:** {p['detected_on']}\n"
        message += f"⏰ **Action Date:** {p['predicted_action_date']}\n"
        message += f"📊 **Prediction:** {p['prediction']}\n"
        message += f"🎲 **Avg Confidence:** {p['avg_confidence']:.1f}%\n"
        message += f"📈 **Companies Affected:** {p['total_companies']} stocks\n\n"
        
        message += "**🎯 TOP OPPORTUNITIES (by confidence):**\n"
        for i, company in enumerate(p['companies'][:5], 1):
            message += f"{i}. **{company['ticker']}**\n"
            message += f"   • Target Level: **${company['target_level']:.2f}** (Friday Low)\n"
            message += f"   • Current: ${company['friday_close']:.2f}\n"
            message += f"   • Potential Move: **{company['potential_move_pct']:.1f}%** downside\n"
            message += f"   • Confidence: **{company['confidence']:.0f}%**\n"
            message += f"   • Pattern Strength: {company['high_diff_pct']:.1f}% high difference\n"
        
        message += f"\n_Full list: {', '.join([c['ticker'] for c in p['companies'][5:10]])}_\n\n"
    
    # Wednesday-Monday Pattern
    if patterns.get('wednesday_monday'):
        p = patterns['wednesday_monday']
        message += "**📅 Pattern #2: Wednesday-Monday Analysis**\n"
        message += f"🎯 **Pattern Detected:** {p['description']}\n"
        message += f"📆 **Detected On:** {p['detected_on']}\n"
        message += f"⏰ **Action Date:** {p['predicted_action_date']}\n"
        message += f"📊 **Prediction:** {p['prediction']}\n"
        message += f"🎲 **Avg Confidence:** {p['avg_confidence']:.1f}%\n"
        message += f"📈 **Companies Affected:** {p['total_companies']} stocks\n\n"
        
        message += "**🎯 TOP OPPORTUNITIES (by confidence):**\n"
        for i, company in enumerate(p['companies'][:5], 1):
            message += f"{i}. **{company['ticker']}**\n"
            message += f"   • Target Level: **${company['target_level']:.2f}** (Wednesday Low)\n"
            message += f"   • Current: ${company['wednesday_close']:.2f}\n"
            message += f"   • Potential Move: **{company['potential_move_pct']:.1f}%** downside\n"
            message += f"   • Confidence: **{company['confidence']:.0f}%**\n"
            message += f"   • Pattern Strength: {company['high_diff_pct']:.1f}% high difference\n"
        
        message += f"\n_Full list: {', '.join([c['ticker'] for c in p['companies'][5:10]])}_\n\n"
    
    # Trading Recommendations
    message += "**💡 TRADING IMPLICATIONS:**\n"
    if patterns.get('friday_thursday'):
        message += "• Set alerts at Friday low levels for potential entry points\n"
        message += "• Watch for Monday morning weakness\n"
        message += "• Consider protective puts or tighter stops\n"
    if patterns.get('wednesday_monday'):
        message += "• Monitor Wednesday lows for Thursday revisit\n"
        message += "• Mid-week reversal opportunity\n"
        message += "• Consider scaling into positions at target levels\n"
    
    message += "\n⚠️ _Historical Accuracy: ~70-75% | Always use proper risk management_\n"
    
    return message


def build_pattern_message_french(patterns: Dict) -> str:
    """Build French pattern message."""
    message = "**🔍 ANALYSE SITUATIONNELLE DU MARCHÉ**\n"
    message += "_Signaux de Trading Basés sur les Patterns pour l'Aide à la Décision_\n\n"
    
    # Friday-Thursday Pattern
    if patterns.get('friday_thursday'):
        p = patterns['friday_thursday']
        message += "**📅 Pattern #1: Analyse Vendredi-Jeudi**\n"
        message += f"🎯 **Pattern Détecté:** {p['description']}\n"
        message += f"📆 **Détecté Le:** {p['detected_on']}\n"
        message += f"⏰ **Date d'Action:** {p['predicted_action_date']}\n"
        message += f"📊 **Prédiction:** {p['prediction']}\n"
        message += f"🎲 **Confiance Moyenne:** {p['avg_confidence']:.1f}%\n"
        message += f"📈 **Sociétés Affectées:** {p['total_companies']} actions\n\n"
        
        message += "**🎯 MEILLEURES OPPORTUNITÉS (par confiance):**\n"
        for i, company in enumerate(p['companies'][:5], 1):
            message += f"{i}. **{company['ticker']}**\n"
            message += f"   • Niveau Cible: **${company['target_level']:.2f}** (Plus bas vendredi)\n"
            message += f"   • Actuel: ${company['friday_close']:.2f}\n"
            message += f"   • Mouvement Potentiel: **{company['potential_move_pct']:.1f}%** baisse\n"
            message += f"   • Confiance: **{company['confidence']:.0f}%**\n"
            message += f"   • Force du Pattern: {company['high_diff_pct']:.1f}% différence des hauts\n"
        
        message += f"\n_Liste complète: {', '.join([c['ticker'] for c in p['companies'][5:10]])}_\n\n"
    
    # Wednesday-Monday Pattern
    if patterns.get('wednesday_monday'):
        p = patterns['wednesday_monday']
        message += "**📅 Pattern #2: Analyse Mercredi-Lundi**\n"
        message += f"🎯 **Pattern Détecté:** {p['description']}\n"
        message += f"📆 **Détecté Le:** {p['detected_on']}\n"
        message += f"⏰ **Date d'Action:** {p['predicted_action_date']}\n"
        message += f"📊 **Prédiction:** {p['prediction']}\n"
        message += f"🎲 **Confiance Moyenne:** {p['avg_confidence']:.1f}%\n"
        message += f"📈 **Sociétés Affectées:** {p['total_companies']} stocks\n\n"
        
        message += "**🎯 MEILLEURES OPPORTUNITÉS (par confiance):**\n"
        for i, company in enumerate(p['companies'][:5], 1):
            message += f"{i}. **{company['ticker']}**\n"
            message += f"   • Niveau Cible: **${company['target_level']:.2f}** (Plus bas mercredi)\n"
            message += f"   • Actuel: ${company['wednesday_close']:.2f}\n"
            message += f"   • Mouvement Potentiel: **{company['potential_move_pct']:.1f}%** baisse\n"
            message += f"   • Confiance: **{company['confidence']:.0f}%**\n"
            message += f"   • Force du Pattern: {company['high_diff_pct']:.1f}% différence des hauts\n"
        
        message += f"\n_Liste complète: {', '.join([c['ticker'] for c in p['companies'][5:10]])}_\n\n"
    
    # Trading Recommendations
    message += "**💡 IMPLICATIONS POUR LE TRADING:**\n"
    if patterns.get('friday_thursday'):
        message += "• Définir des alertes aux niveaux bas du vendredi pour points d'entrée potentiels\n"
        message += "• Surveiller la faiblesse du lundi matin\n"
        message += "• Considérer des puts protecteurs ou stops plus serrés\n"
    if patterns.get('wednesday_monday'):
        message += "• Surveiller les plus bas du mercredi pour revisite jeudi\n"
        message += "• Opportunité de renversement en milieu de semaine\n"
        message += "• Considérer l'entrée progressive aux niveaux cibles\n"
    
    message += "\n⚠️ _Précision Historique: ~70-75% | Toujours utiliser une gestion des risques appropriée_\n"
    
    return message


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
    """Build formatted Discord message with enhanced pattern analysis."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    language = os.getenv("LANGUAGE", "EN").upper()
    
    messages = []
    
    # ENGLISH VERSION
    if language in ["EN", "BOTH"]:
        message_en = f"**📊 S&P 500 Market Update - {timestamp}**\n\n"
        
        # Top Gainers - ENHANCED FORMAT
        if top_movers.get('gainers'):
            message_en += "**🚀 Top Gainers (>75th percentile)**\n"
            for ticker, change in top_movers['gainers'][:5]:
                if not pd.isna(change) and ticker in market_data:
                    # Use new enhanced format
                    formatted = format_enhanced_ticker(ticker, market_data[ticker])
                    message_en += f"• {formatted}\n"
            message_en += "\n"
        
        # Top Losers - ENHANCED FORMAT  
        if top_movers.get('losers'):
            message_en += "**📉 Top Losers (>75th percentile)**\n"
            for ticker, change in top_movers['losers'][:5]:
                if not pd.isna(change) and ticker in market_data:
                    # Use new enhanced format
                    formatted = format_enhanced_ticker(ticker, market_data[ticker])
                    message_en += f"• {formatted}\n"
            message_en += "\n"
        
        # Enhanced Pattern Analysis
        if patterns and patterns.get('active_patterns'):
            pattern_section = build_enhanced_pattern_message(patterns, "EN")
            message_en += pattern_section
        
        message_en += "\n_Automated by GitHub Actions_"
        messages.append(message_en)
    
    # FRENCH VERSION
    if language in ["FR", "BOTH"]:
        message_fr = f"**📊 Actualisation Marché S&P 500 - {timestamp}**\n\n"
        
        # Top Gainers
        if top_movers.get('gainers'):
            message_fr += "**🚀 Plus Fortes Hausses (>75e centile)**\n"
            for ticker, change in top_movers['gainers'][:5]:
                if not pd.isna(change):
                    message_fr += f"• {ticker}: **+{change:.2f}%**\n"
            message_fr += "\n"
        
        # Top Losers
        if top_movers.get('losers'):
            message_fr += "**📉 Plus Fortes Baisses (>75e centile)**\n"
            for ticker, change in top_movers['losers'][:5]:
                if not pd.isna(change):
                    message_fr += f"• {ticker}: **{change:.2f}%**\n"
            message_fr += "\n"
        
        # Enhanced Pattern Analysis
        if patterns and patterns.get('active_patterns'):
            pattern_section = build_enhanced_pattern_message(patterns, "FR")
            message_fr += pattern_section
        
        message_fr += "\n_Automatisé par GitHub Actions_"
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
    logger.info("S&P 500 Discord Monitor - GitHub Actions")
    logger.info("="*60)
    
    try:
        # Validate configuration
        if not DISCORD_WEBHOOK_URL:
            raise ValueError("DISCORD_WEBHOOK_URL environment variable not set")

        # Load company names
        global COMPANY_NAMES
        COMPANY_NAMES = get_company_names()
        
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
        
        # Step 5: Enhanced pattern analysis
        patterns = analyze_enhanced_patterns(historical_data)
        
        # Step 6: Build and send message
        message = build_message(top_movers, patterns)

        # Track prices for next update
        track_price_history(market_data)
        
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
