import json
import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from mGPT.callback import build_callbacks
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def print_table(title, metrics, logger=None):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

    logger.info(metrics) if logger else None


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # Logger
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))

    # Output dir
    model_name = cfg.model.target.split('.')[-2].lower()
    output_dir = Path(
        os.path.join(cfg.FOLDER, model_name, cfg.NAME, "samples_" + cfg.TIME))
    if cfg.TEST.SAVE_PREDICTIONS:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to {str(output_dir)}")

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase="test")
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, logger, phase="test")
    else:
        logger.warning("No checkpoints provided!!!")

    # Calculate metrics
    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES

    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
        metrics = trainer.test(model, datamodule=datamodule)[0]
        if "TM2TMetrics" in metrics_type and cfg.model.params.task == "t2m" and cfg.model.params.stage != 'vae':
            # mm meteics
            logger.info(f"Evaluating MultiModality - Replication {i}")
            datamodule.mm_mode(True)
            mm_metrics = trainer.test(model, datamodule=datamodule)[0]
            # metrics.update(mm_metrics)
            metrics.update(mm_metrics)
            datamodule.mm_mode(False)
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}

    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval

    print_table(f"Mean Metrics", all_metrics_new, logger=logger)
    all_metrics_new.update(all_metrics)

    # Save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
