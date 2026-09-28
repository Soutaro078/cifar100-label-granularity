# CIFAR-100 ラベル粒度の比較

CIFAR-100 の Fine（100クラス）と Coarse（20クラス）で事前学習した ResNet-18 を、少数サンプルのファインチューニングで比べる実験です。

## ディレクトリ

```
cifar100-label-granularity/
├── data/                  # CIFAR-100
├── outputs/
│   ├── models/            # 事前学習の重み (.pth)
│   └── figures/           # グラフ
├── src/
│   ├── models/            # ResNet-18（32×32 用）
│   ├── utils/             # パス、設定、前処理、ラベル読み込み
│   ├── pretrain.py
│   ├── finetune.py
│   ├── plot_results.py
│   └── run_experiment_grid.py
├── configs/               # pretrain.yaml, finetune.yaml
├── notebooks/
├── requirements.txt
└── README.md
```

`src/check_cifar100.py`、`src/check_resnet18.py`、`src/train_random_test.py` はデータの中身と学習ループの動作確認用です。

## セットアップ

```bash
pip install -r requirements.txt
```

データは `data/` に置いてあります。無い場合は `pretrain.py` 実行時に torchvision が `data/` へダウンロードします。

## 事前学習

設定は `configs/pretrain.yaml` です。エポック数は 200、学習率は `CosineAnnealingLR`（`T_max` はエポック数）です。バッチサイズは 128。学習時の拡張は `RandomCrop(32, padding=4)` と `RandomHorizontalFlip`、正規化は CIFAR-100 の平均・標準偏差です。

Fine（100クラス）と Coarse（20クラス）は別々に学習し、重みは次の名前で保存します。

```bash
python src/pretrain.py --mode fine
python src/pretrain.py --mode coarse
```

- Fine事前学習モデル: `outputs/models/resnet18_pretrain_fine.pth`
- Coarse事前学習モデル: `outputs/models/resnet18_pretrain_coarse.pth`

モデルは 32×32 用です。第1畳み込みは 3×3・stride 1・padding 1、maxpool は Identity です。この仕様は `src/models/resnet.py` の `build_cifar_resnet18` で組み立て時に確認します。

CLI で YAML を上書きできます。

```bash
python src/pretrain.py --mode fine --epochs 200 --batch_size 128 --lr 0.1
```

## ファインチューニングと比較

初期化は Fine 事前学習、Coarse 事前学習、ランダムの3通りです。学習画像は拡張せず、評価と同じ正規化だけを使います。エポック数の既定は `configs/finetune.yaml` の 10 で、事前学習の 200 とは別です。

```bash
python src/finetune.py --init_mode fine --target_group trees --samples_per_class 10
python src/run_experiment_grid.py --target_group vehicles_1
python src/plot_results.py --target_group vehicles_1
```

グラフは `outputs/figures/` に保存します。

## Weights & Biases

事前学習とファインチューニングは、プロジェクト `cifar100-label-granularity` に記録します。先に `wandb login` を済ませてから実行してください。記録を止めるときは `--no_wandb` を付けます。

事前学習の run 名は `pretrain-fine` と `pretrain-coarse` です。横軸はエポックです。Accuracy はパーセントです。検証曲線は CIFAR-100 の test split を使います。この値で学習を早めに止めることはしません。

Fine 学習（100クラス）では、そのエポックの Fine の Loss / Accuracy に加え、予測を公式の coarse 対応へ写した Coarse（20クラス）の Loss / Accuracy も記録します。メトリクス名は `train/fine_accuracy`、`val/fine_accuracy`、`train/coarse_accuracy`、`val/coarse_accuracy` などです。Coarse 学習の出力は 20 クラスなので、記録されるのは Coarse の Loss / Accuracy です。2本の run を並べると、Coarse Accuracy の曲線を比べられます。

ファインチューニングは `finetune/train_loss`、`finetune/train_accuracy`、`finetune/test_accuracy` を、初期化（fine / coarse / random）と対象グループが分かる run 名で記録します。
