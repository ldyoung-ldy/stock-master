#!/usr/bin/env python3
"""
同步分析结果到飞书
功能：
1. 读取 Excel 持仓并进行实时技术分析
2. 将持仓盈亏同步到「持仓管理」表
3. 将技术指标和建议同步到「数据表」（技术分析表）
"""

import sys
import os
from datetime import datetime

# 设置环境变量
os.environ['FEISHU_CONFIG_PATH'] = '/Users/solaeter/Documents/ldyoung/投资理财/美股管理Agent/stock-master/feishu_config.json'

# 添加脚本目录
sys.path.insert(0, '/Users/solaeter/Documents/ldyoung/投资理财/美股管理Agent/stock-master/scripts')

import indicators
import beginner_analyzer
import portfolio
from data_fetcher import DataFetcher
from feishu_sync import FeishuBitable, sync_holding, sync_stock_signal

def main():
    print("=" * 70)
    print("🚀 开始同步分析结果到飞书")
    print("=" * 70)

    # 1. 加载持仓
    print("\n[1/4] 读取持仓数据...")
    try:
        portfolio_data = portfolio.read_portfolio()
        holdings = portfolio_data['holdings']
        if not holdings:
            print("❌ 持仓为空")
            return
        print(f"✅ 找到 {len(holdings)} 个持有标的")
    except Exception as e:
        print(f"❌ 读取持仓失败: {e}")
        return

    # 2. 初始化数据获取器和飞书
    fetcher = DataFetcher()
    bitable = FeishuBitable()
    
    # 定义表 ID
    HOLDINGS_TABLE_ID = "tblh8LfgGYq3sVl3"  # 持仓管理
    ANALYSIS_TABLE_ID = "tbl0oP7vDHy9cvOa"  # 数据表 (技术分析)

    print("\n[2/4] 正在分析并同步数据...")
    
    for h in holdings:
        ticker = h['ticker']
        shares = h['shares']
        avg_cost = h['avg_cost']
        
        print(f"\n🔍 正在分析 {ticker}...")
        
        # 获取数据
        df = fetcher.get_stock_data(ticker, period="3mo")
        if df is None or df.empty:
            print(f"  ⚠️  无法获取 {ticker} 数据，跳过...")
            continue
            
        current_price = df['Close'].iloc[-1]
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        # --- 计算技术指标 ---
        rsi_val = indicators.calculate_rsi(close_prices)
        macd_res = indicators.calculate_macd(close_prices)
        kdj_res = indicators.calculate_kdj(high_prices, low_prices, close_prices)
        bb_res = indicators.calculate_bollinger_bands(close_prices)
        
        macd_hist = macd_res.get('histogram', 0) if macd_res else 0
        prev_macd_hist = macd_res.get('prev_histogram', 0) if macd_res else 0
        bb_upper = bb_res.get('upper', 0) if bb_res else 0
        bb_middle = bb_res.get('middle', 0) if bb_res else 0
        bb_lower = bb_res.get('lower', 0) if bb_res else 0
        kdj_signal = kdj_res.get('signal', '') if kdj_res else ''
        
        # 生成建议
        rec = beginner_analyzer.generate_trading_recommendation(
            ticker=ticker,
            current_price=current_price,
            rsi=rsi_val or 50,
            macd_histogram=macd_hist,
            prev_macd_histogram=prev_macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            kdj_signal=kdj_signal
        )
        
        # --- A. 同步到「持仓管理」 ---
        print(f"  📤 同步基本信息到「持仓管理」...")
        try:
            pnl = (current_price - avg_cost) * shares
            pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
            
            # 判断市场
            if '.HK' in ticker: market = '港股'
            elif '.SS' in ticker or '.SZ' in ticker: market = 'A股'
            else: market = '美股'
            
            h_data = {
                'ticker': ticker,
                'name': h.get('name', ''),
                'quantity': shares,
                'cost_price': avg_cost,
                'current_price': current_price,
                'profit_amount': pnl,
                'profit_ratio': pnl_pct,
                'market': market,
                'buy_date': h.get('buy_date', ''),
                'note': h.get('notes', '')
            }
            sync_holding(bitable, h_data, HOLDINGS_TABLE_ID)
            print("    ✅ 成功")
        except Exception as e:
            print(f"    ❌ 失败: {e}")

        # --- B. 同步到「技术分析/数据表」 ---
        print(f"  📤 同步分析结果到「数据表」...")
        try:
            # 兼容性处理
            action = rec.action if hasattr(rec, 'action') else rec.get('action', 'HOLD')
            score = rec.score if hasattr(rec, 'score') else rec.get('score', 0)
            reasons = rec.reasons if hasattr(rec, 'reasons') else rec.get('reasons', [])
            
            s_data = {
                'ticker': ticker,
                'name': h.get('name', ''),
                'current_price': current_price,
                'score': score,
                'action': action,
                'rsi': rsi_val or 50,
                'macd_signal': macd_res.get('interpretation', '') if macd_res else '',
                'kdj_signal': kdj_res.get('interpretation', '') if kdj_res else '',
                'reasons': reasons,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            sync_stock_signal(bitable, s_data, ANALYSIS_TABLE_ID)
            print("    ✅ 成功")
        except Exception as e:
            print(f"    ❌ 失败: {e}")

    print("\n" + "=" * 70)
    print(f"✨ 同步任务完成 - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

if __name__ == '__main__':
    main()
