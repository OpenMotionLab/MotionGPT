from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
from mGPT.config import instantiate_from_config

class TM2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "matching, fid, and diversity scores"
        self.top_k = top_k
        self.R_size = R_size
        self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        # Matching scores
        if self.text:
            self.add_state("Matching_score",
                            default=torch.tensor(0.0),
                            dist_reduce_fx="sum")
            self.add_state("gt_Matching_score",
                            default=torch.tensor(0.0),
                            dist_reduce_fx="sum")
            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
            for k in range(1, top_k + 1):
                self.add_state(
                    f"R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.add_state(
                    f"gt_R_precision_top_{str(k)}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
            self.metrics.extend(self.Matching_metrics)

        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # Chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

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
            cfg.METRIC.TM2T.t2m_path, dataname, "text_mot_match/model/finest.tar"),
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

    @torch.no_grad()
    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count_seq)

        all_genmotions = torch.cat(self.recmotion_embeddings,
                                   axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]

        # Compute text related metrics
        if self.text:
            all_texts = torch.cat(self.text_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]
            # Compute r-precision
            assert count_seq > self.R_size
            top_k_mat = torch.zeros((self.top_k, ))
            for i in range(count_seq // self.R_size):
                # [bs=32, 1*256]
                group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 1*256]
                group_motions = all_genmotions[i * self.R_size:(i + 1) *
                                               self.R_size]
                # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
                # [bs=32, 32]
                dist_mat = euclidean_distance_matrix(
                    group_texts, group_motions).nan_to_num()
                # print(dist_mat[:5])
                self.Matching_score += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax,
                                             top_k=self.top_k).sum(axis=0)

            R_count = count_seq // self.R_size * self.R_size
            metrics["Matching_score"] = self.Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

            # Compute r-precision with gt
            assert count_seq > self.R_size
            top_k_mat = torch.zeros((self.top_k, ))
            for i in range(count_seq // self.R_size):
                # [bs=32, 1*256]
                group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
                # [bs=32, 1*256]
                group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                              self.R_size]
                # [bs=32, 32]
                dist_mat = euclidean_distance_matrix(
                    group_texts, group_motions).nan_to_num()
                # match score
                self.gt_Matching_score += dist_mat.trace()
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax,
                                             top_k=self.top_k).sum(axis=0)
            metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
            for k in range(self.top_k):
                metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions,
                                                      self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(
            all_gtmotions, self.diversity_times)

        # Reset
        self.reset()

        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor,
               feats_rst: Tensor,
               lengths_ref: List[int],
               lengths_rst: List[int],
               word_embs: Tensor = None,
               pos_ohot: Tensor = None,
               text_lengths: Tensor = None):

        self.count += sum(lengths_ref)
        self.count_seq += len(lengths_ref)

        # T2m motion encoder
        align_idx = np.argsort(lengths_ref)[::-1].copy()
        feats_ref = feats_ref[align_idx]
        lengths_ref = np.array(lengths_ref)[align_idx]
        gtmotion_embeddings = self.get_motion_embeddings(
            feats_ref, lengths_ref)
        cache = [0] * len(lengths_ref)
        for i in range(len(lengths_ref)):
            cache[align_idx[i]] = gtmotion_embeddings[i:i + 1]
        self.gtmotion_embeddings.extend(cache)

        align_idx = np.argsort(lengths_rst)[::-1].copy()
        feats_rst = feats_rst[align_idx]
        lengths_rst = np.array(lengths_rst)[align_idx]
        recmotion_embeddings = self.get_motion_embeddings(
            feats_rst, lengths_rst)
        cache = [0] * len(lengths_rst)
        for i in range(len(lengths_rst)):
            cache[align_idx[i]] = recmotion_embeddings[i:i + 1]
        self.recmotion_embeddings.extend(cache)

        # T2m text encoder
        if self.text:
            text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)
            text_embeddings = torch.flatten(text_emb, start_dim=1).detach()
            self.text_embeddings.append(text_embeddings)

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        m_lens = torch.tensor(lengths)
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")
        m_lens = m_lens // self.cfg.DATASET.HUMANML3D.UNIT_LEN
        mov = self.t2m_moveencoder(feats[..., :-4]).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        return torch.flatten(emb, start_dim=1).detach()
