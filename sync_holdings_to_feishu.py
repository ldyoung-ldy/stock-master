#!/usr/bin/env python3
"""
同步 Excel 持仓到飞书
"""

import sys
import os

# 设置环境变量
os.environ['FEISHU_CONFIG_PATH'] = '/Users/solaeter/Documents/ldyoung/投资理财/美股管理Agent/stock-master/feishu_config.json'

sys.path.insert(0, '/Users/solaeter/Documents/ldyoung/投资理财/美股管理Agent/stock-master/scripts')

from portfolio import read_portfolio
from feishu_sync import FeishuBitable, sync_holding

def main():
    # 读取 Excel 持仓
    print('📊 读取 Excel 持仓数据...')
    portfolio = read_portfolio('/Users/solaeter/Documents/ldyoung/投资理财/美股管理Agent/stock-master/my_portfolio.xlsx')
    holdings = portfolio['holdings']
    print(f'  找到 {len(holdings)} 条持仓记录')

    # 连接飞书
    print('\n☁️ 连接飞书多维表格...')
    bitable = FeishuBitable()

    # 获取持仓表 ID
    tables = bitable.list_tables()
    holdings_table_id = None
    for t in tables:
        if t['name'] == '持仓管理':
            holdings_table_id = t['table_id']
            break

    if not holdings_table_id:
        print('❌ 找不到持仓管理表')
        sys.exit(1)

    print(f'  持仓表 ID: {holdings_table_id}')

    # 同步每条持仓
    print('\n🔄 同步持仓数据...')
    success = 0
    for h in holdings:
        try:
            # 判断市场
            ticker = h['ticker']
            if '.HK' in ticker:
                market = '港股'
            elif '.SS' in ticker or '.SZ' in ticker:
                market = 'A股'
            else:
                market = '美股'
            
            holding_data = {
                'ticker': ticker,
                'name': h.get('name', ''),
                'quantity': h['shares'],
                'cost_price': h['avg_cost'],
                'current_price': h.get('current_price') or 0,
                'profit_amount': h.get('profit_loss') or 0,
                'profit_ratio': h.get('profit_loss_pct') or 0,
                'market': market,
                'buy_date': h.get('buy_date', ''),
                'note': h.get('notes', '')
            }
            
            sync_holding(bitable, holding_data, holdings_table_id)
            print(f'  ✓ {ticker} ({h.get("name", "")}): {h["shares"]}股 @ ${h["avg_cost"]}')
            success += 1
        except Exception as e:
            print(f'  ✗ {h["ticker"]}: {e}')

    print(f'\n✅ 同步完成! 成功: {success}/{len(holdings)}')
    print('\n💡 打开飞书多维表格查看「持仓管理」表')

if __name__ == '__main__':
    main()
