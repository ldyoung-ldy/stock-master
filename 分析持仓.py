#!/usr/bin/env python3
"""
股票持仓分析脚本
自动读取持仓文件，获取实时数据并进行技术分析
"""

import sys
import os
from datetime import datetime
import pandas as pd

# 添加脚本目录到搜索路径
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, script_dir)

import indicators
import beginner_analyzer
import portfolio
from data_fetcher import DataFetcher

def analyze_portfolio():
    print("=" * 70)
    print("📋 我的美股持仓深度分析")
    print("=" * 70)
    
    # 1. 加载持仓
    print("\n[第一步] 读取持仓数据...")
    try:
        portfolio_data = portfolio.read_portfolio()
        holdings = portfolio_data['holdings']
        if not holdings:
            print("❌ 持仓为空，请检查 my_portfolio.xlsx")
            return
        print(f"✅ 找到 {len(holdings)} 个持有标的")
    except Exception as e:
        print(f"❌ 读取持仓失败: {e}")
        return

    # 2. 初始化数据获取器
    fetcher = DataFetcher()
    
    results = []
    total_cost = 0
    total_market_value = 0
    
    # 3. 逐个分析
    print("\n[第二步] 开始扫描各个标的...")
    for h in holdings:
        ticker = h['ticker']
        shares = h['shares']
        avg_cost = h['avg_cost']
        
        print(f"\n🔍 正在分析 {ticker} (持有 {shares} 股, 成本 ${avg_cost:.2f})...")
        
        # 获取历史数据 (3个月)
        df = fetcher.get_stock_data(ticker, period="3mo")
        
        if df is None or df.empty:
            print(f"  ⚠️  无法获取 {ticker} 数据，跳过...")
            continue
            
        # 获取基础指标
        current_price = df['Close'].iloc[-1]
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        # 计算财务
        holding_cost = shares * avg_cost
        market_value = shares * current_price
        pnl = market_value - holding_cost
        pnl_pct = (pnl / holding_cost) * 100 if holding_cost > 0 else 0
        
        total_cost += holding_cost
        total_market_value += market_value
        
        # 技术分析
        rsi_val = indicators.calculate_rsi(close_prices)
        macd_res = indicators.calculate_macd(close_prices)
        kdj_res = indicators.calculate_kdj(high_prices, low_prices, close_prices)
        bb_res = indicators.calculate_bollinger_bands(close_prices)
        
        # 准备高级分析参数
        macd_hist = macd_res.get('histogram', 0) if macd_res else 0
        prev_macd_hist = macd_res.get('prev_histogram', 0) if macd_res else 0
        bb_upper = bb_res.get('upper', 0) if bb_res else 0
        bb_middle = bb_res.get('middle', 0) if bb_res else 0
        bb_lower = bb_res.get('lower', 0) if bb_res else 0
        kdj_signal = kdj_res.get('signal', '') if kdj_res else ''
        
        # 使用新手分析器生成建议 (v3.4 签名)
        rec = beginner_analyzer.generate_trading_recommendation(
            ticker=ticker,
            current_price=current_price,
            rsi=rsi_val or 50,
            macd_histogram=macd_hist,
            prev_macd_histogram=prev_macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            kdj_signal=kdj_signal,
            kdj_k=kdj_res.get('k') if kdj_res else None,
            kdj_d=kdj_res.get('d') if kdj_res else None,
            kdj_j=kdj_res.get('j') if kdj_res else None
        )
        
        # 兼容性处理：如果返回的是 TradingSignal 对象
        if hasattr(rec, 'action'):
            action = rec.action
            score = rec.score
            reasons = rec.reasons
        else:
            action = rec.get('action', '观察')
            score = rec.get('score', 0)
            reasons = rec.get('reasons', [])

        results.append({
            'ticker': ticker,
            'name': h.get('name', ticker),
            'shares': shares,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'rsi': rsi_val,
            'recommendation': action,
            'score': score,
            'reasons': reasons
        })
        
        print(f"  价格: ${current_price:.2f} | 盈亏: {pnl_pct:+.2f}% | 建议: {action}")

    # 4. 生成汇总报告
    print("\n" + "=" * 70)
    print("📊 持仓分析报告汇总")
    print("=" * 70)
    
    print(f"{'代码':<8} {'现价':<10} {'盈亏%':<10} {'RSI':<8} {'建议':<10}")
    print("-" * 70)
    
    for r in results:
        pnl_str = f"{r['pnl_pct']:+.2f}%"
        rsi_str = f"{r['rsi']:.1f}" if r['rsi'] else "N/A"
        print(f"{r['ticker']:<8} ${r['current_price']:<10.2f} {pnl_str:<10} {rsi_str:<8} {r['recommendation']:<10}")

    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"💰 账户总额概览")
    print(f"  总投入:     ${total_cost:,.2f}")
    print(f"  当前市值:   ${total_market_value:,.2f}")
    print(f"  总盈亏:     ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    print("=" * 70)
    
    # 5. 焦点建议
    print("\n💡 关键操作建议:")
    for r in results:
        if abs(r['score']) >= 5:
            emoji = "🚨" if r['score'] <= -5 else "✨"
            print(f"  {emoji} {r['ticker']}: {r['recommendation']}")
            for reason in r['reasons'][:2]:
                print(f"     - {reason}")

    print("\n分析时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 70)

if __name__ == "__main__":
    analyze_portfolio()
