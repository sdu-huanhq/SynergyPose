import logging
import os
import os.path as osp
import mmcv
import ref
from detectron2.data import DatasetCatalog
from core.modeling.datasets import (
    lm_pbr,
    lmo_bop_test,
)


cur_dir = osp.dirname(osp.abspath(__file__))
__all__ = [
    "register_dataset",
    "register_datasets",
    "register_datasets_in_cfg",
    "get_available_datasets",
]
_DSET_MOD_NAMES = [
    "lm_pbr",
    "lmo_bop_test",
]

logger = logging.getLogger(__name__)


def register_dataset(mod_name, dset_name, data_cfg=None):
    register_func = eval(mod_name)
    register_func.register_with_name_cfg(dset_name, data_cfg)


def get_available_datasets(mod_name):
    name = eval(mod_name)
    return name.get_available_datasets()


def register_datasets_in_cfg(cfg):
    for split in [
        "TRAIN",
        "TEST",
    ]:
        for name in cfg.DATASETS.get(split, []):
            if name in DatasetCatalog.list(): 
                continue
            registered = False
            for _mod_name in _DSET_MOD_NAMES:  
                if name in get_available_datasets(_mod_name):
                    register_dataset(_mod_name, name, data_cfg=None)
                    registered = True
                    break
            if not registered:
                assert "DATA_CFG" in cfg and name in cfg.DATA_CFG, "no cfg.DATA_CFG.{}".format(name)
                assert osp.exists(cfg.DATA_CFG[name])
                data_cfg = mmcv.load(cfg.DATA_CFG[name])
                mod_name = data_cfg.pop("mod_name", None)
                assert mod_name in _DSET_MOD_NAMES, mod_name
                register_dataset(mod_name, name, data_cfg)


def register_datasets(dataset_names):
    for name in dataset_names:
        if name in DatasetCatalog.list():
            continue
        registered = False
        for _mod_name in _DSET_MOD_NAMES:
            if name in get_available_datasets(_mod_name):
                register_dataset(_mod_name, name, data_cfg=None)
                registered = True
                break

        if not registered:
            raise ValueError(f"dataset {name} is not defined")
