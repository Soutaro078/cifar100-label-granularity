class GranularityMeters:
    """学習しているヘッド（Fine なら 100、Coarse なら 20）の Loss / Accuracy だけを蓄積する。

    引数は Python の数値に限る。ロジットテンソルや logsumexp は受け取らない。
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss_value: float, correct: int, batch_size: int) -> None:
        self.loss_sum += float(loss_value) * int(batch_size)
        self.correct += int(correct)
        self.total += int(batch_size)

    def summarize(self) -> dict:
        if self.total == 0:
            raise ValueError("集計するバッチがありません")
        return {
            "loss": self.loss_sum / self.total,
            "accuracy": 100.0 * self.correct / self.total,
        }


def pretrain_wandb_metrics(mode: str, train: dict, val: dict, epoch: int, lr: float) -> dict:
    """エポックごとの記録。Accuracy はパーセント。

    記録するのは、その run のヘッドが直接出している損失と正解率だけ。
    Fine の予測を 20 クラスへ写す計算はしない。
    """
    metrics = {
        "epoch": epoch,
        "lr": lr,
        "train/loss": train["loss"],
        "train/accuracy": train["accuracy"],
        "val/loss": val["loss"],
        "val/accuracy": val["accuracy"],
    }
    if mode == "fine":
        metrics.update(
            {
                "train/fine_loss": train["loss"],
                "train/fine_accuracy": train["accuracy"],
                "val/fine_loss": val["loss"],
                "val/fine_accuracy": val["accuracy"],
            }
        )
    elif mode == "coarse":
        metrics.update(
            {
                "train/coarse_loss": train["loss"],
                "train/coarse_accuracy": train["accuracy"],
                "val/coarse_loss": val["loss"],
                "val/coarse_accuracy": val["accuracy"],
            }
        )
    else:
        raise ValueError(f"mode は 'fine' または 'coarse' です: {mode}")
    return metrics
