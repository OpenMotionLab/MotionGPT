import os
import numpy as np
import torch
import logging
from pathlib import Path
from pytorch_lightning import LightningModule
from os.path import join as pjoin
from collections import OrderedDict
from mGPT.metrics import BaseMetrics
from mGPT.config import get_obj_from_str


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configure_metrics()

        # Ablation
        self.test_step_outputs = []
        self.times = []
        self.rep_i = 0

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step("test", batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        dico.update(self.loss_log_dict('val'))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self):
        # Log metrics
        dico = self.metrics_log_dict()
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        self.save_npy(self.test_step_outputs)
        self.rep_i = self.rep_i + 1
        # Free up the memory
        self.test_step_outputs.clear()

    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        
        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict['metrics.' + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict['_losses.' + k] = v

        for k, v in state_dict.items():
            if '_losses' not in k and 'Metrics' not in k:
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        super().load_state_dict(new_state_dict, strict)

    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch)
        }

    def loss_log_dict(self, split: str):
        losses = self._losses['losses_' + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):

        # For TM2TMetrics MM
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
            metrics_dicts = ['MMMetrics']
        else:
            metrics_dicts = self.hparams.metrics_dict

        # Compute all metrics
        metrics_log_dict = {}
        for metric in metrics_dicts:
            metrics_dict = getattr(
                self.metrics,
                metric).compute(sanity_flag=self.trainer.sanity_checking)
            metrics_log_dict.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })

        return metrics_log_dict
    
    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=self.parameters(), **self.hparams.cfg.TRAIN.OPTIM.params)

        # Scheduler
        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    def save_npy(self, outputs):
        cfg = self.hparams.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.target.split('.')[-2].lower()),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        if cfg.TEST.SAVE_PREDICTIONS:
            lengths = [i[1] for i in outputs]
            outputs = [i[0] for i in outputs]

            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        data = self.trainer.datamodule.test_dataset.data_dict[
                            keyid]

                        motion = torch.tensor(data['motion'],
                                              device=outputs[i].device)
                        motion = self.datamodule.normalize(motion)
                        length = data['length']
                        text_list = data['text']
                        gen_joints = outputs[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}.npy"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
                        npypath = output_dir / f"{keyid}_gt.npy"
                        joints = self.feats2joints(motion).cpu().numpy()
                        np.save(npypath, joints)

                        with open(output_dir / f"{keyid}.txt", "a") as f:
                            for text in text_list:
                                f.write(f"{text['caption']}\n")

            elif cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
                keyids = range(len(self.trainer.datamodule.test_dataset))
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu()
                        gen_joints = gen_joints.permute(2, 0,
                                                        1)[:lengths[i][bid],
                                                           ...].numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{self.rep_i}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
