import wandb


def init_wandb(
    project_name: str,
    run_name: str,
    dataset_name: str,
    config: dict,
    group: str | None = None,
    tags: list[str] | None = None,
):
    """Initialize a wandb run.

    Args:
        project_name: wandb project name.
        run_name: display name for this run.
        dataset_name: dataset identifier (e.g. "movielens1m").
        config: flat dict of hyperparameters to log.
        group: optional wandb run group — seed replicas of one config share a
            group so wandb aggregates them (mean/std) out of the box.
        tags: optional list of wandb tags.
    """
    run = wandb.init(
        entity=None, # Omitted for anonymity. Set to the relevant wandb entity.
        project=project_name,
        name=run_name,
        group=group,
        tags=tags,
        reinit="finish_previous",
    )
    wandb.config.update(
        {
            "dataset_name": dataset_name,
            **config,
        }
    )
    return run
