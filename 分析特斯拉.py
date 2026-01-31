#!/usr/bin/env python3
"""
特斯拉 (TSLA) 股票分析脚本
使用本地计算和实时数据分析
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import indicators
import beginner_analyzer

print("=" * 70)
print("🚗 特斯拉 (TSLA) 股票技术分析")
print("=" * 70)
print()

# 获取特斯拉数据
ticker = "TSLA"
print(f"正在获取 {ticker} 股票数据...")

try:
    # 获取股票对象
    stock = yf.Ticker(ticker)
    
    # 获取基本信息
    info = stock.info
    
    # 获取历史数据 (3个月)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    df = stock.history(period="3mo")
    
    if df.empty:
        print("❌ 无法获取股票数据,可能是网络问题或API限制")
        print("💡 建议: 稍后再试或检查网络连接")
        sys.exit(1)
    
    # 显示基本信息
    print("\n" + "=" * 70)
    print("📊 基本信息")
    print("=" * 70)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100
    
    print(f"公司名称: {info.get('longName', 'Tesla, Inc.')}")
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
    rsi_result = indicators.calculate_rsi(df)
    if rsi_result:
        rsi_val = rsi_result.get('rsi')
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
    macd_result = indicators.calculate_macd(df)
    if macd_result:
        macd_val = macd_result.get('macd')
        signal = macd_result.get('signal_line')
        macd_signal = macd_result.get('signal', '')
        
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
    kdj_result = indicators.calculate_kdj(df)
    if kdj_result:
        k_val = kdj_result.get('K')
        d_val = kdj_result.get('D')
        j_val = kdj_result.get('J')
        kdj_signal = kdj_result.get('signal', '')
        
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
    bb_result = indicators.calculate_bollinger_bands(df)
    if bb_result:
        upper = bb_result.get('upper')
        middle = bb_result.get('middle')
        lower = bb_result.get('lower')
        
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
            for pattern in candlestick[:3]:  # 显示前3个
                name = pattern.get('name', '')
                signal = pattern.get('signal', '')
                strength = pattern.get('strength', '')
                emoji = '🟢' if signal == '看涨' else '🔴' if signal == '看跌' else '⚪'
                print(f"  {emoji} {name} - {signal} ({strength})")
        
        if chart:
            print("\n【趋势形态】")
            for pattern in chart[:3]:  # 显示前3个
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
    print("• 特斯拉股价波动较大,注意风险控制")
    
    print("\n" + "=" * 70)
    print("分析完成 - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

except Exception as e:
    print(f"\n❌ 分析过程出错: {e}")
    print("\n可能的原因:")
    print("  1. Yahoo Finance API 暂时限流")
    print("  2. 网络连接问题")
    print("  3. 股票代码错误")
    print("\n💡 建议:")
    print("  • 等待15-30分钟后重试")
    print("  • 检查网络连接")
    print("  • 或在 Claude 对话中直接说: '分析 TSLA 股票'")
    
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
