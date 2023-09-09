from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric

from .utils import *


# motion reconstruction metric
class MRMetrics(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Reconstructions'
        self.jointstype = jointstype
        self.align_root = align_root
        self.force_in_meter = force_in_meter

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("MPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("PAMPJPE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("ACCEL",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        # todo
        # self.add_state("ROOT", default=torch.tensor([0.0]), dist_reduce_fx="sum")

        self.MR_metrics = ["MPJPE", "PAMPJPE", "ACCEL"]

        # All metric
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):
        if self.force_in_meter:
            # different jointstypes have different scale factors
            # if self.jointstype == 'mmm':
            #     factor = 1000.0
            # elif self.jointstype == 'humanml3d':
            #     factor = 1000.0 * 0.75 / 480
            factor = 1000.0
        else:
            factor = 1.0

        count = self.count
        count_seq = self.count_seq
        mr_metrics = {}
        mr_metrics["MPJPE"] = self.MPJPE / count * factor
        mr_metrics["PAMPJPE"] = self.PAMPJPE / count * factor
        # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        # n-2 for each sequences
        mr_metrics["ACCEL"] = self.ACCEL / (count - 2 * count_seq) * factor
        
        # Reset
        self.reset()
        
        return mr_metrics

    def update(self, joints_rst: Tensor, joints_ref: Tensor,
               lengths: List[int]):
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # avoid cuda error of DDP in pampjpe
        rst = joints_rst.detach().cpu()
        ref = joints_ref.detach().cpu()

        # align root joints index
        if self.align_root and self.jointstype in ['mmm', 'humanml3d']:
            align_inds = [0]
        else:
            align_inds = None

        for i in range(len(lengths)):
            self.MPJPE += torch.sum(
                calc_mpjpe(rst[i], ref[i], align_inds=align_inds))
            self.PAMPJPE += torch.sum(calc_pampjpe(rst[i], ref[i]))
            self.ACCEL += torch.sum(calc_accel(rst[i], ref[i]))
