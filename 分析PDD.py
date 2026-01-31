#!/usr/bin/env python3
"""
拼多多 (PDD) 股票分析脚本
使用本地计算和实时数据分析（带缓存和智能重试）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import pandas as pd
from datetime import datetime
import indicators
import beginner_analyzer
from data_fetcher import DataFetcher

print("=" * 70)
print("🛍️  拼多多 (PDD) 股票技术分析")
print("=" * 70)
print()

# 获取数据
ticker = "PDD"
print(f"正在获取 {ticker} 股票数据...")

try:
    # 使用新的数据获取器（带缓存和重试）
    fetcher = DataFetcher()
    
    # 获取历史数据 (3个月)
    df = fetcher.get_stock_data(ticker, period="3mo")
    
    if df is None or df.empty:
        print("❌ 无法获取股票数据,可能是网络问题或API限制")
        print("💡 建议: 稍后再试或检查网络连接")
        sys.exit(1)
    
    # 获取基本信息（带重试）
    info = fetcher.get_stock_info(ticker)
    if info is None:
        info = {}
    
    # 显示基本信息
    print("\n" + "=" * 70)
    print("📊 基本信息")
    print("=" * 70)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100
    
    print(f"公司名称: {info.get('longName', 'PDD Holdings Inc.')}")
    print(f"股票代码: {ticker}")
    print(f"当前价格: ${current_price:.2f}")
    print(f"涨跌额: ${change:+.2f} ({change_pct:+.2f}%)")
    print(f"52周最高: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52周最低: ${info.get('fiftyTwoWeekLow', 'N/A')}")
    print(f"市值: ${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "市值: N/A")
    
    # 计算技术指标
    print("\n" + "=" * 70)
    print("📈 技术指标分析")
    print("=" * 70)
    
    # RSI
    close_prices = df['Close'].values
    rsi_val = indicators.calculate_rsi(close_prices)
    if rsi_val:
        print(f"\n【RSI - 相对强弱指数】")
        print(f"  数值: {rsi_val:.2f}")
        if rsi_val < 30:
            print(f"  信号: 超卖 🔵")
            print(f"  解读: 像大甩卖,股价可能被低估,可以考虑买入")
        elif rsi_val > 70:
            print(f"  信号: 超买 🔴")
            print(f"  解读: 被抢购一空,涨得有点多,小心回调")
        else:
            print(f"  信号: 正常区间 ⚪")
            print(f"  解读: 价格在合理范围内波动")
    
    # MACD
    macd_result = indicators.calculate_macd(close_prices)
    if macd_result and 'error' not in macd_result:
        macd_val = macd_result.get('macd_line', 0)
        signal = macd_result.get('signal_line', 0)
        macd_signal = macd_result.get('interpretation', '')
        
        print(f"\n【MACD - 趋势动量指标】")
        print(f"  MACD: {macd_val:.2f}")
        print(f"  信号线: {signal:.2f}")
        
        if '金叉' in macd_signal:
            print(f"  信号: 金叉 🟢")
            print(f"  解读: 像踩油门加速,买入信号")
        elif '死叉' in macd_signal:
            print(f"  信号: 死叉 🔴")
            print(f"  解读: 像松油门减速,卖出警告")
        else:
            print(f"  信号: {macd_signal}")
    
    # KDJ
    high_prices = df['High'].values
    low_prices = df['Low'].values
    kdj_result = indicators.calculate_kdj(high_prices, low_prices, close_prices)
    if kdj_result and 'error' not in kdj_result:
        k_val = kdj_result.get('k', 0)
        d_val = kdj_result.get('d', 0)
        j_val = kdj_result.get('j', 0)
        kdj_signal = kdj_result.get('interpretation', '')
        
        print(f"\n【KDJ - 随机指标】")
        print(f"  K值: {k_val:.2f}")
        print(f"  D值: {d_val:.2f}")
        print(f"  J值: {j_val:.2f}")
        
        if '金叉' in kdj_signal or j_val < 0:
            print(f"  信号: 看涨 🟢")
            print(f"  解读: 绿灯亮了,短期买入机会")
        elif '死叉' in kdj_signal or j_val > 100:
            print(f"  信号: 看跌 🔴")
            print(f"  解读: 红灯亮了,短期卖出信号")
        else:
            print(f"  信号: {kdj_signal}")
    
    # 布林带
    bb_result = indicators.calculate_bollinger_bands(close_prices)
    if bb_result and 'error' not in bb_result:
        upper = bb_result.get('upper', 0)
        middle = bb_result.get('middle', 0)
        lower = bb_result.get('lower', 0)
        
        print(f"\n【布林带 - 波动区间】")
        print(f"  上轨: ${upper:.2f}")
        print(f"  中轨: ${middle:.2f}")
        print(f"  下轨: ${lower:.2f}")
        print(f"  当前: ${current_price:.2f}")
        
        if current_price < lower:
            print(f"  信号: 跌破下轨 🔵")
            print(f"  解读: 橡皮筋拉太长,可能反弹")
        elif current_price > upper:
            print(f"  信号: 突破上轨 🔴")
            print(f"  解读: 涨过头了,可能回落")
        else:
            print(f"  信号: 在正常区间 ⚪")
    
    # 形态识别
    print("\n" + "=" * 70)
    print("📐 形态识别")
    print("=" * 70)
    
    patterns_result = indicators.analyze_patterns(df)
    if patterns_result:
        candlestick = patterns_result.get('candlestick_patterns', [])
        chart = patterns_result.get('chart_patterns', [])
        
        if candlestick:
            print("\n【K线形态】")
            for pattern in candlestick[:3]:
                name = pattern.get('name', '')
                signal = pattern.get('signal', '')
                strength = pattern.get('strength', '')
                emoji = '🟢' if signal == '看涨' else '🔴' if signal == '看跌' else '⚪'
                print(f"  {emoji} {name} - {signal} ({strength})")
        
        if chart:
            print("\n【趋势形态】")
            for pattern in chart[:3]:
                name = pattern.get('name', '')
                signal = pattern.get('signal', '')
                emoji = '🟢' if signal == '看涨' else '🔴' if signal == '看跌' else '⚪'
                print(f"  {emoji} {name} - {signal}")
    
    # 综合评分和建议
    print("\n" + "=" * 70)
    print("💡 综合交易建议")
    print("=" * 70)
    
    # 构建分析数据
    analysis_data = {
        'rsi': rsi_val if rsi_result else None,
        'macd': macd_result,
        'kdj': kdj_result,
        'bollinger': bb_result,
        'patterns': patterns_result,
        'current_price': current_price
    }
    
    # 生成交易建议
    recommendation = beginner_analyzer.generate_trading_recommendation(analysis_data)
    
    if recommendation:
        score = recommendation.get('score', 0)
        action = recommendation.get('action', '观望')
        position = recommendation.get('position_size', '-')
        reasons = recommendation.get('reasons', [])
        
        print(f"\n综合评分: {score:+d} 分 (范围: -10 到 +10)")
        print(f"交易建议: {action}")
        if position != '-':
            print(f"建议仓位: {position}")
        
        print(f"\n主要理由:")
        for i, reason in enumerate(reasons[:5], 1):
            print(f"  {i}. {reason}")
    
    # 风险提示
    print("\n" + "=" * 70)
    print("⚠️  风险提示")
    print("=" * 70)
    print("• 本分析仅供参考,不构成投资建议")
    print("• 股市有风险,投资需谨慎")
    print("• 建议分批建仓,设置止损")
    print("• 拼多多股价波动较大,注意风险控制")
    
    print("\n" + "=" * 70)
    print("分析完成 - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

except Exception as e:
    print(f"\n❌ 分析过程出错: {e}")
    import traceback
    traceback.print_exc()
