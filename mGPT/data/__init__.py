import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BASEDataModule(pl.LightningDataModule):
    def __init__(self, collate_fn):
        super().__init__()

        self.dataloader_options = {"collate_fn": collate_fn}
        self.persistent_workers = True
        self.is_mm = False

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.DatasetEval(**sample_params)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(split=self.cfg.TRAIN.SPLIT,
                                               **self.hparams)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.EVAL.SPLIT
            self._val_dataset = self.DatasetEval(**params)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            # self._test_dataset = self.DatasetEval(split=self.cfg.TEST.SPLIT,
            #                                       **self.hparams)
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.TEST.SPLIT
            self._test_dataset = self.DatasetEval( **params)
        return self._test_dataset

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.TRAIN.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TRAIN.NUM_WORKERS
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            persistent_workers=True,
            **dataloader_options,
        )

    def predict_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def val_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def test_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )
