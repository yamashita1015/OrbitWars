# OrbitWars Agent

[Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) コンペティション用のエージェントです。

## 概要

惑星を占領しながら宇宙を制圧する1v1 / 4人FFA形式のシミュレーションゲームです。  
ルールベースのヒューリスティックエージェントを実装しています（機械学習なし）。

## 戦略

### ベース：Flow Diff スコアリング
各ターン、自分の惑星から敵・中立惑星への攻撃候補を列挙し、「この艦隊を送った場合に18ターン後の生産量がどれだけ増えるか」をスコア化。ROI閾値（1.5）を超えた攻撃のみ実行します。

### 改善1：ETA-aware Reinforcement Risk
長距離攻撃は敵が増援を送る時間があるため、飛行時間（ETA）に応じて必要艦隊数を上乗せします。遠距離の採算の合わない攻撃を自動的に抑制します。

### 改善2：Multi-size Candidates
従来は常に最大艦隊数で攻撃候補を生成していましたが、0.5倍・1.0倍の2種類を生成するよう変更。1つのソース惑星が1ターンに複数のターゲットを攻撃できるようになりました。

## ファイル構成

```
main.py                  # エージェント本体
orbit_lite/              # ヘルパーライブラリ（slawekbiel作）
test_local.py            # ローカルテスト用スクリプト
tune_params.py           # パラメータ調整スクリプト
kaggle_submission.ipynb  # Kaggle提出用ノートブック
docs_rules.md            # ゲームルール詳細
docs_strategy.md         # 戦略の詳細説明
```

## ローカルでの実行

### 環境構築

```bash
python -m venv .venv
source .venv/bin/activate
pip install kaggle-environments torch numpy --no-deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### テスト実行

```bash
# 自己対戦（20ゲーム）
python test_local.py --games 20 --opponent self

# スナイパーエージェントと対戦
python test_local.py --games 20 --opponent sniper
```

### 提出ファイルの作成

```bash
tar -czf submission.tar.gz main.py orbit_lite/
```

## 技術スタック

- Python 3.14
- PyTorch（テンソル演算のみ、機械学習なし）
- kaggle-environments
