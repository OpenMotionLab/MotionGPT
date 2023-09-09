from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
import os
from mGPT.config import instantiate_from_config

class MMMetrics(Metric):
    full_state_update = True

    def __init__(self, cfg, dataname='humanml3d', mm_num_times=10, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MultiModality scores"
        self.cfg = cfg
        self.dataname = dataname
        self.mm_num_times = mm_num_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = ["MultiModality"]
        self.add_state("MultiModality",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")

        # chached batches
        self.add_state("mm_motion_embeddings", default=[], dist_reduce_fx=None)

        # T2M Evaluator
        self._get_t2m_evaluator(cfg)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_textencoder)
        self.t2m_moveencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_moveencoder)
        self.t2m_motionencoder = instantiate_from_config(cfg.METRIC.TM2T.t2m_motionencoder)

        # load pretrianed
        if self.dataname == "kit":
            dataname = "kit"
        else:
            dataname = "t2m"
        t2m_checkpoint = torch.load(os.path.join(
            cfg.METRIC.TM2T.t2m_path, dataname,
            "text_mot_match/model/finest.tar"),
                                    map_location="cpu")

        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        all_mm_motions = torch.cat(self.mm_motion_embeddings,
                                   axis=0).cpu().numpy()
        metrics['MultiModality'] = calculate_multimodality_np(
            all_mm_motions, self.mm_num_times)

        # Reset
        self.reset()

        return {**metrics}

    def update(
        self,
        feats_rst: Tensor,
        lengths_rst: List[int],
    ):
        self.count += sum(lengths_rst)
        self.count_seq += len(lengths_rst)

        align_idx = np.argsort(lengths_rst)[::-1].copy()
        feats_rst = feats_rst[align_idx]
        lengths_rst = np.array(lengths_rst)[align_idx]
        recmotion_embeddings = self.get_motion_embeddings(
            feats_rst, lengths_rst)
        cache = [0] * len(lengths_rst)
        for i in range(len(lengths_rst)):
            cache[align_idx[i]] = recmotion_embeddings[i:i + 1]

        mm_motion_embeddings = torch.cat(cache, axis=0).unsqueeze(0)
        # self.mm_motion_embeddings.extend(cache)
        # print(mm_motion_embeddings.shape)
        # # store all mm motion embeddings
        self.mm_motion_embeddings.append(mm_motion_embeddings)

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        m_lens = torch.tensor(lengths)
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        mov = self.t2m_moveencoder(feats[..., :-4]).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        return torch.flatten(emb, start_dim=1).detach()
