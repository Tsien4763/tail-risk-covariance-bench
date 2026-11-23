# Tail-Risk Covariance Benchmark（尾部风险协方差基准）

[English](README.md)

全样本协方差矩阵在市场下跌尾部会低估多少组合波动，又该如何在杜绝前向填充与幸存者偏差的前提下评测尾部条件协方差估计器？

本仓库将严格日度收益面板构建与尾部风险协方差评测打包在一起。数据入库层强制校验日期单调唯一性、禁用收盘价前向填充、屏蔽 Fama-French 缺失哨兵值并记录 SHA-256 溯源信息；估计层则在同一面板上对比样本协方差、线性收缩协方差与左尾条件协方差的最小方差组合表现。

- `src/prepare_panels.py`：严格日期索引 CSV 读取、缺失哨兵值屏蔽与 SHA-256 质量报告。
- `src/covariance_bench.py`：样本、线性收缩与左尾条件协方差估计器及解析最小方差权重。

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` 生成含 10% 压力交易日的 250 天合成多资产收益面板，调用 `prepare_panels.dateindexed` 载入并对比各估计器的条件数与尾部方差。

## 范围说明

本仓库为合成收益面板上的公开 MVP，基于 Kenneth French 公开数据与 ETF 面板的完整实证脚本按需提供。
