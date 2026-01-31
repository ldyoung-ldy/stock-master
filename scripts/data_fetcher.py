#!/usr/bin/env python3
"""
统一数据获取模块 v1.0

功能：
- 本地 SQLite 缓存，减少重复请求
- 智能重试机制（指数退避）
- 多数据源自动切换（Yahoo → FMP → 本地计算）

使用示例：
    from data_fetcher import DataFetcher
    fetcher = DataFetcher()
    df = fetcher.get_stock_data("AAPL", period="3mo")
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np


class StockDataCache:
    """本地 SQLite 缓存管理器"""
    
    def __init__(self, cache_dir: str = None, ttl_minutes: int = 15):
        """
        初始化缓存
        
        Args:
            cache_dir: 缓存目录，默认为 scripts 目录
            ttl_minutes: 缓存有效期（分钟）
        """
        if cache_dir is None:
            cache_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.db_path = os.path.join(cache_dir, "stock_cache.db")
        self.ttl_minutes = ttl_minutes
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    cache_key TEXT PRIMARY KEY,
                    ticker TEXT,
                    period TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quote_cache (
                    ticker TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _make_key(self, ticker: str, period: str) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{ticker}_{period}".encode()).hexdigest()
    
    def get_price_data(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """获取缓存的价格数据"""
        cache_key = self._make_key(ticker, period)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data, created_at FROM price_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row:
                data, created_at = row
                created_time = datetime.fromisoformat(created_at)
                
                # 检查是否过期
                if datetime.now() - created_time < timedelta(minutes=self.ttl_minutes):
                    try:
                        df = pd.read_json(data)
                        print(f"  💾 从缓存读取 {ticker} 数据")
                        return df
                    except Exception:
                        pass
        
        return None
    
    def set_price_data(self, ticker: str, period: str, df: pd.DataFrame):
        """保存价格数据到缓存"""
        cache_key = self._make_key(ticker, period)
        data = df.to_json(date_format='iso')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO price_cache 
                   (cache_key, ticker, period, data, created_at) 
                   VALUES (?, ?, ?, ?, ?)""",
                (cache_key, ticker, period, data, datetime.now().isoformat())
            )
            conn.commit()
    
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """获取缓存的实时报价"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data, created_at FROM quote_cache WHERE ticker = ?",
                (ticker,)
            )
            row = cursor.fetchone()
            
            if row:
                data, created_at = row
                created_time = datetime.fromisoformat(created_at)
                
                # 报价缓存 5 分钟
                if datetime.now() - created_time < timedelta(minutes=5):
                    return json.loads(data)
        
        return None
    
    def set_quote(self, ticker: str, quote: Dict):
        """保存实时报价到缓存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO quote_cache 
                   (ticker, data, created_at) VALUES (?, ?, ?)""",
                (ticker, json.dumps(quote), datetime.now().isoformat())
            )
            conn.commit()
    
    def clear(self, ticker: str = None):
        """清理缓存"""
        with sqlite3.connect(self.db_path) as conn:
            if ticker:
                conn.execute("DELETE FROM price_cache WHERE ticker = ?", (ticker,))
                conn.execute("DELETE FROM quote_cache WHERE ticker = ?", (ticker,))
            else:
                conn.execute("DELETE FROM price_cache")
                conn.execute("DELETE FROM quote_cache")
            conn.commit()
    
    def cleanup_expired(self):
        """清理过期缓存"""
        threshold = (datetime.now() - timedelta(hours=24)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM price_cache WHERE created_at < ?", 
                (threshold,)
            )
            conn.execute(
                "DELETE FROM quote_cache WHERE created_at < ?", 
                (threshold,)
            )
            conn.commit()


class RetryHandler:
    """智能重试处理器（指数退避）"""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 5.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
    
    def execute(self, func, *args, **kwargs):
        """
        执行函数，失败时自动重试
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值，或在重试耗尽后返回 None
        """
        last_error = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # 检查是否是限流错误
                if "rate limit" in error_str or "too many" in error_str or "429" in error_str:
                    if attempt < self.max_retries - 1:
                        print(f"  ⏳ 触发限流，等待 {delay:.0f} 秒后重试 ({attempt + 1}/{self.max_retries})...")
                        time.sleep(delay)
                        delay *= 2  # 指数退避
                    continue
                else:
                    # 其他错误直接抛出
                    raise e
        
        # 重试耗尽
        print(f"  ⚠️ 重试 {self.max_retries} 次后仍失败: {last_error}")
        return None


class DataFetcher:
    """
    统一数据获取接口
    
    特性：
    - 自动缓存（SQLite）
    - 智能重试（指数退避）
    - 多数据源切换
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化数据获取器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        cache_ttl = self.config.get('data_sources', {}).get('cache_ttl_minutes', 15)
        max_retries = self.config.get('data_sources', {}).get('max_retries', 3)
        retry_delay = self.config.get('data_sources', {}).get('retry_delay_seconds', 5)
        
        self.cache = StockDataCache(ttl_minutes=cache_ttl)
        self.retry_handler = RetryHandler(max_retries=max_retries, initial_delay=retry_delay)
        
        self._yf = None
        self._fmp_key = self.config.get('data_sources', {}).get('fmp_api_key', '')
        self._polygon_key = self.config.get('data_sources', {}).get('polygon_api_key', '')
    
    def _load_config(self, config_path: str = None) -> Dict:
        """加载配置文件"""
        if config_path is None:
            # 默认配置文件路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _ensure_yfinance(self):
        """延迟导入 yfinance"""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
    
    def _fetch_from_yahoo(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """从 Yahoo Finance 获取数据"""
        self._ensure_yfinance()
        
        stock = self._yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        return df
    
    def _fetch_from_fmp(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """从 Financial Modeling Prep 获取数据（使用新版 stable API）"""
        if not self._fmp_key:
            return None
        
        try:
            import requests
            
            # 使用新版 stable API 端点
            url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
            params = {
                'symbol': ticker,
                'apikey': self._fmp_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"  ⚠️ FMP API 返回状态码: {response.status_code}")
                return None
            
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                
                # 根据 period 过滤数据
                days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
                days = days_map.get(period, 90)
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df.index >= cutoff_date.strftime('%Y-%m-%d')]
                
                # 重命名列以匹配 yfinance 格式
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # 只保留需要的列
                cols_to_keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                return df[cols_to_keep]
        
        except Exception as e:
            print(f"  ⚠️ FMP API 错误: {e}")
        
        return None
    
    def _fetch_from_polygon(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """从 Polygon.io 获取数据（免费 5 次/分钟，支持所有美股包括中国 ADR）"""
        if not self._polygon_key:
            return None
        
        try:
            import requests
            
            # 计算日期范围
            days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
            days = days_map.get(period, 90)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Polygon.io Aggregates (Bars) API
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apiKey': self._polygon_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"  ⚠️ Polygon API 返回状态码: {response.status_code}")
                return None
            
            data = response.json()
            
            # Polygon 免费账户返回 'DELAYED' 状态，也是有效数据
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data:
                results = data['results']
                df = pd.DataFrame(results)
                
                # 转换时间戳为日期
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                
                # 重命名列以匹配 yfinance 格式
                df = df.rename(columns={
                    'o': 'Open',
                    'h': 'High',
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume'
                })
                
                cols_to_keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                return df[cols_to_keep]
        
        except Exception as e:
            print(f"  ⚠️ Polygon API 错误: {e}")
        
        return None
    
    def get_stock_data(self, ticker: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """
        获取股票历史数据（带缓存和多数据源切换）
        
        Args:
            ticker: 股票代码
            period: 时间周期 (1mo, 3mo, 6mo, 1y)
        
        Returns:
            包含 OHLCV 数据的 DataFrame，失败返回 None
        """
        ticker = ticker.upper()
        
        # 1. 尝试从缓存获取
        cached = self.cache.get_price_data(ticker, period)
        if cached is not None:
            return cached
        
        print(f"  🌐 正在获取 {ticker} 股票数据...")
        
        # 2. 尝试 Yahoo Finance（带重试）
        df = self.retry_handler.execute(self._fetch_from_yahoo, ticker, period)
        
        if df is not None and not df.empty:
            print(f"  ✓ Yahoo Finance 获取成功")
            self.cache.set_price_data(ticker, period, df)
            return df
        
        # 3. 尝试 FMP 备用源
        if self._fmp_key:
            print(f"  ➜ 尝试 FMP 备用数据源...")
            df = self._fetch_from_fmp(ticker, period)
            
            if df is not None and not df.empty:
                print(f"  ✓ FMP API 获取成功")
                self.cache.set_price_data(ticker, period, df)
                return df
        
        # 4. 尝试 Polygon.io 备用源（支持中国 ADR）
        if self._polygon_key:
            print(f"  ➜ 尝试 Polygon.io 备用数据源...")
            df = self._fetch_from_polygon(ticker, period)
            
            if df is not None and not df.empty:
                print(f"  ✓ Polygon API 获取成功")
                self.cache.set_price_data(ticker, period, df)
                return df
        
        print(f"  ❌ 无法获取 {ticker} 数据")
        return None
    
    def get_realtime_quote(self, ticker: str) -> Optional[Dict]:
        """
        获取实时报价（带缓存）
        
        Args:
            ticker: 股票代码
        
        Returns:
            报价字典，包含 price, open, high, low, volume 等
        """
        ticker = ticker.upper()
        
        # 检查缓存
        cached = self.cache.get_quote(ticker)
        if cached:
            return cached
        
        self._ensure_yfinance()
        
        try:
            stock = self._yf.Ticker(ticker)
            info = stock.fast_info
            
            quote = {
                'ticker': ticker,
                'price': float(info.last_price) if hasattr(info, 'last_price') else None,
                'open': float(info.open) if hasattr(info, 'open') else None,
                'high': float(info.day_high) if hasattr(info, 'day_high') else None,
                'low': float(info.day_low) if hasattr(info, 'day_low') else None,
                'volume': int(info.last_volume) if hasattr(info, 'last_volume') else None,
                'previous_close': float(info.previous_close) if hasattr(info, 'previous_close') else None,
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache.set_quote(ticker, quote)
            return quote
        
        except Exception as e:
            print(f"  ⚠️ 获取报价失败: {e}")
            return None
    
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        获取股票基本信息（带重试）
        
        Args:
            ticker: 股票代码
        
        Returns:
            信息字典
        """
        ticker = ticker.upper()
        self._ensure_yfinance()
        
        def fetch_info():
            stock = self._yf.Ticker(ticker)
            return stock.info
        
        return self.retry_handler.execute(fetch_info)
    
    def clear_cache(self, ticker: str = None):
        """
        清理缓存
        
        Args:
            ticker: 指定股票代码，为 None 时清理所有缓存
        """
        self.cache.clear(ticker)
        print(f"  🗑️ 缓存已清理" + (f" ({ticker})" if ticker else ""))


# 便捷函数
_default_fetcher = None

def get_fetcher() -> DataFetcher:
    """获取默认数据获取器实例"""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = DataFetcher()
    return _default_fetcher


def fetch_stock_data(ticker: str, period: str = '3mo') -> Optional[pd.DataFrame]:
    """便捷函数：获取股票数据"""
    return get_fetcher().get_stock_data(ticker, period)


def fetch_quote(ticker: str) -> Optional[Dict]:
    """便捷函数：获取实时报价"""
    return get_fetcher().get_realtime_quote(ticker)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("DataFetcher 模块测试")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    # 测试获取数据
    print("\n[测试] 获取 AAPL 3个月数据...")
    df = fetcher.get_stock_data("AAPL", "3mo")
    if df is not None:
        print(f"  ✓ 成功获取 {len(df)} 条记录")
        print(f"  最新收盘价: ${df['Close'].iloc[-1]:.2f}")
    else:
        print("  ✗ 获取失败")
    
    # 再次获取（测试缓存）
    print("\n[测试] 再次获取 AAPL（应从缓存读取）...")
    df2 = fetcher.get_stock_data("AAPL", "3mo")
    if df2 is not None:
        print(f"  ✓ 缓存读取成功")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
