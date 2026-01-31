#!/usr/bin/env python3
"""
Stock Master 功能测试脚本
测试本地计算功能是否正常工作
"""

import sys
import os

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

print("=" * 60)
print("Stock Master 功能测试")
print("=" * 60)

# 测试 1: 导入核心模块
print("\n[测试 1] 检查核心模块导入...")
try:
    import indicators
    import beginner_analyzer
    import portfolio
    print("  ✓ 所有核心模块导入成功")
except ImportError as e:
    print(f"  ✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试 2: 检查依赖包
print("\n[测试 2] 检查依赖包...")
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import openpyxl
    print("  ✓ yfinance 版本:", yf.__version__)
    print("  ✓ pandas 版本:", pd.__version__)
    print("  ✓ numpy 版本:", np.__version__)
    print("  ✓ openpyxl 版本:", openpyxl.__version__)
except ImportError as e:
    print(f"  ✗ 依赖包缺失: {e}")
    sys.exit(1)

# 测试 3: 测试本地指标计算
print("\n[测试 3] 测试本地指标计算功能...")
try:
    # 创建测试数据
    test_data = pd.DataFrame({
        'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                  111, 110, 112, 114, 113, 115, 117, 116, 118, 120],
        'High': [101, 103, 102, 104, 106, 105, 107, 109, 108, 110,
                 112, 111, 113, 115, 114, 116, 118, 117, 119, 121],
        'Low': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108,
                110, 109, 111, 113, 112, 114, 116, 115, 117, 119],
        'Volume': [1000000] * 20
    })
    
    # 测试 RSI 计算
    rsi_result = indicators.calculate_rsi(test_data)
    if rsi_result and 'rsi' in rsi_result:
        print(f"  ✓ RSI 计算成功: {rsi_result['rsi']:.2f}")
    else:
        print("  ✗ RSI 计算失败")
    
    # 测试 MACD 计算
    macd_result = indicators.calculate_macd(test_data)
    if macd_result and 'macd' in macd_result:
        print(f"  ✓ MACD 计算成功")
    else:
        print("  ✗ MACD 计算失败")
    
    # 测试布林带计算
    bb_result = indicators.calculate_bollinger_bands(test_data)
    if bb_result and 'upper' in bb_result:
        print(f"  ✓ 布林带计算成功")
    else:
        print("  ✗ 布林带计算失败")
        
    # 测试 KDJ 计算
    kdj_result = indicators.calculate_kdj(test_data)
    if kdj_result and 'K' in kdj_result:
        print(f"  ✓ KDJ 计算成功: K={kdj_result['K']:.2f}")
    else:
        print("  ✗ KDJ 计算失败")
    
except Exception as e:
    print(f"  ✗ 指标计算出错: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 测试小白分析器
print("\n[测试 4] 测试小白分析器...")
try:
    # 测试 RSI 解读
    rsi_explanation = beginner_analyzer.explain_rsi_simple({'rsi': 75.0, 'signal': '超买'})
    if rsi_explanation:
        print("  ✓ RSI 小白解读功能正常")
    
    # 测试交易建议生成
    test_analysis = {
        'rsi': 75.0,
        'macd': {'signal': '金叉'},
        'kdj': {'signal': '超买'},
        'patterns': []
    }
    recommendation = beginner_analyzer.generate_trading_recommendation(test_analysis)
    if recommendation:
        print(f"  ✓ 交易建议生成成功")
        print(f"    - 评分: {recommendation.get('score', 'N/A')}")
        print(f"    - 建议: {recommendation.get('action', 'N/A')}")
    
except Exception as e:
    print(f"  ✗ 小白分析器出错: {e}")
    import traceback
    traceback.print_exc()

# 测试 5: 检查配置文件
print("\n[测试 5] 检查配置文件...")
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(config_path):
    print(f"  ✓ 配置文件存在: {config_path}")
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"  ✓ 配置文件格式正确")
        print(f"    - 持仓路径: {config.get('portfolio_path', 'N/A')}")
    except Exception as e:
        print(f"  ✗ 配置文件读取失败: {e}")
else:
    print(f"  ⚠ 配置文件不存在,但不影响基本功能")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n使用说明:")
print("  1. 作为 Claude Skill 使用(推荐):")
print("     在对话中直接说: '分析 AAPL 股票'")
print("")
print("  2. 直接运行脚本:")
print("     cd scripts && python3 main.py AAPL")
print("")
print("  3. 查看详细文档:")
print("     查看 使用指南.md 文件")
print("=" * 60)
