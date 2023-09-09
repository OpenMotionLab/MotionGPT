import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def get_path(sample_path: Path, is_amass: bool, gender: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    gender_str = gender + "_" if is_amass else ""
    path = sample_path / f"{fact_str}{gender_str}{split}{extra_str}"
    return path
