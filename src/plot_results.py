import argparse

import matplotlib.pyplot as plt

from utils.paths import FIGURES_DIR

def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 サンプル効率曲線グラフ描画")
    parser.add_argument("--target_group", type=str, default="vehicles_1",
                        choices=["trees", "vehicles_1", "aquatic_mammals"],
                        help="描画するグループ名 ('trees', 'vehicles_1', 'aquatic_mammals')")
    args = parser.parse_args()

    # 各グループの実験データ（サマリー）
    data_dict = {
        'trees': {
            'title': 'Target: Trees (5 classes)',
            'accuracies': {
                'Fine (100 classes)': [47.80, 35.00, 50.40, 54.80],
                'Coarse (20 classes)': [35.80, 29.00, 45.20, 50.20],
                'Random (Baseline)': [29.60, 24.00, 30.20, 35.20]
            }
        },
        'vehicles_1': {
            'title': 'Target: Vehicles_1 (5 classes)',
            'accuracies': {
                'Fine (100 classes)': [49.00, 40.40, 58.00, 64.80],
                'Coarse (20 classes)': [28.80, 37.20, 47.20, 51.00],
                'Random (Baseline)': [30.00, 21.80, 23.40, 32.20]
            }
        },
        'aquatic_mammals': {
            'title': 'Target: Aquatic Mammals (5 classes)',
            'accuracies': {
                'Fine (100 classes)': [38.80, 33.40, 46.80, 50.00],
                'Coarse (20 classes)': [36.20, 33.40, 42.80, 49.60],
                'Random (Baseline)': [31.00, 35.20, 23.20, 42.60]
            }
        }
    }

    if args.target_group not in data_dict:
        raise ValueError(f"エラー: 指定されたグループ '{args.target_group}' のデータがありません。")

    target_data = data_dict[args.target_group]
    samples = [5, 10, 20, 50]

    # グラフのスタイルの設定
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for (model_name, acc_list), marker, color in zip(target_data['accuracies'].items(), markers, colors):
        plt.plot(samples, acc_list, marker=marker, linewidth=2, markersize=8, label=model_name, color=color)

    # グラフの装飾
    plt.title(f'Sample Efficiency Curve ({target_data["title"]})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Training Samples per Class', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(samples, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(15, 70) # 範囲を少し広めに調整
    
    # 凡例とグリッド
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / f"sample_efficiency_curve_{args.target_group}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[完了] グラフを保存しました: {save_path}")

if __name__ == '__main__':
    main()