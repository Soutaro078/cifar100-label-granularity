def wandb_settings(cfg: dict, no_wandb: bool, project_override: str | None) -> tuple[bool, str]:
    enabled = bool(cfg.get("wandb", True)) and not no_wandb
    project = project_override or cfg.get("wandb_project") or "cifar100-label-granularity"
    return enabled, project


def init_wandb(enabled: bool, project: str, name: str, job_type: str, config: dict, group: str):
    if not enabled:
        print("W&B への記録は無効です。")
        return None

    import wandb

    # x_disable_stats: GPU 使用率などの定期採取を止める。
    # この採取は PyTorch と同じ NVIDIA ドライバを叩くため、画面固まりの切り分け対象にする。
    run = wandb.init(
        project=project,
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        settings=wandb.Settings(x_disable_stats=True),
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    url = getattr(run, "url", None)
    print(f"W&B run: {url or run.name}")
    return run


def log_metrics(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
