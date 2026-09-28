import argparse
import re
import subprocess
import sys

from utils.paths import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 複数グループ一括比較実験グリッド")
    parser.add_argument("--target_group", type=str, default="vehicles_1",
                        help="検証する粗いクラスの名前（例: 'trees', 'vehicles_1', 'aquatic_mammals' など）")
    args = parser.parse_args()

    # 比較する条件の定義
    init_modes = ['fine', 'coarse', 'random']
    sample_sizes = [5, 10, 20, 50]
    # ファインチューニング側のエポック。事前学習の 200 エポックとは別。
    epochs = 5

    results = []

    print("=" * 60)
    print(f" 🚀 対象グループ 【 {args.target_group} 】 の一括比較実験を開始します")
    print("=" * 60)
    
    total_runs = len(init_modes) * len(sample_sizes)
    current = 0

    for mode in init_modes:
        for samples in sample_sizes:
            current += 1
            print(f"\n[{current}/{total_runs}] 実行中: init_mode = {mode.upper()}, samples_per_class = {samples}枚")
            
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "finetune.py"),
                "--init_mode",
                mode,
                "--target_group",
                args.target_group,
                "--samples_per_class",
                str(samples),
                "--epochs",
                str(epochs),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 標準出力からテスト正解率 (Accuracy) を抽出
            match = re.search(r'テストデータ正解率 \(Accuracy\): ([\d\.]+)%', result.stdout)
            if match:
                acc = float(match.group(1))
                results.append({'mode': mode, 'samples': samples, 'accuracy': acc})
                print(f"  -> 完了! 正解率 (Accuracy): {acc:.2f}%")
            else:
                print(f"  -> エラー発生または正解率の抽出に失敗しました。")
                print(result.stdout[-200:])

    # 最後に綺麗な一覧表（マトリクス形式）で結果を表示
    print("\n" + "="*60)
    print(f" 📊 【実験結果サマリー一覧】 対象グループ: {args.target_group} (テスト正解率: %)")
    print("="*60)
    print(f"{'サンプル数/クラス':<15} | {'Fine':<10} | {'Coarse':<10} | {'Random':<10}")
    print("-" * 60)

    for samples in sample_sizes:
        row_accs = {}
        for mode in init_modes:
            found = [r['accuracy'] for r in results if r['mode'] == mode and r['samples'] == samples]
            row_accs[mode] = f"{found[0]:.2f}%" if found else "N/A"
        
        print(f"{samples:<15} | {row_accs.get('fine','N/A'):<10} | {row_accs.get('coarse','N/A'):<10} | {row_accs.get('random','N/A'):<10}")
    print("="*60)
    print(f"グループ '{args.target_group}' のデータ収集が完了しました！")

if __name__ == "__main__":
    main()