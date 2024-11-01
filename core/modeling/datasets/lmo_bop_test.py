import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref

from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class LMO_BOP_TEST_Dataset(object):
    """lmo bop test splits."""

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.dataset_root = data_cfg.get("dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test_bop"))
        assert osp.exists(self.dataset_root), self.dataset_root

        self.ann_file = data_cfg["ann_file"] 

        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/lmo/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)

        self.cat_ids = [cat_id for cat_id, obj_name in ref.lmo_full.id2obj.items() if obj_name in self.objs]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj, obj_id in enumerate(self.objs))

    def __call__(self):
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  
        targets = mmcv.load(self.ann_file)

        scene_im_ids = [(item["scene_id"], item["im_id"]) for item in targets]
        
        i, j = 0, 0
        proposal_per_img = {}
        for item in sorted(set(scene_im_ids)):
            proposals = scene_im_ids.count(item)
            j += proposals
            proposals_obj = []
            for item_ in targets[i:j]:
                proposals_obj.append(item_["obj_id"])
                i += 1
            proposal_per_img[item[1]]=proposals_obj
        
        scene_im_ids = sorted(list(set(scene_im_ids)))
            
        gt_dicts = {}
        gt_info_dicts = {}
        cam_dicts = {}
        for scene_id, im_id in scene_im_ids:
            scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
            if scene_id not in gt_dicts:
                gt_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            if scene_id not in gt_info_dicts:
                gt_info_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))  # bbox_obj, bbox_visib
            if scene_id not in cam_dicts:
                cam_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_camera.json"))

        for scene_id, int_im_id in tqdm(scene_im_ids):
            str_im_id = str(int_im_id)
            scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")

            gt_dict = gt_dicts[scene_id]
            gt_info_dict = gt_info_dicts[scene_id]
            cam_dict = cam_dicts[scene_id]

            rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
            assert osp.exists(rgb_path), rgb_path

            depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

            scene_im_id = f"{scene_id}/{int_im_id}"

            K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000

            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "image_id": int_im_id,
                "scene_im_id": scene_im_id,  # for evaluation
                "cam": K,
                "depth_factor": depth_factor,
                "img_type": "real",  # NOTE: has background
            }
            insts = []
            for anno_i, anno in enumerate(gt_dict[str_im_id]):
                obj_id = anno["obj_id"]
                if obj_id not in self.cat_ids:
                    continue
                if obj_id not in proposal_per_img[int_im_id]: 
                    continue 
                cur_label = self.cat2label[obj_id]  # 0-based label
                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                pose = np.hstack([R, t.reshape(3, 1)])
                quat = mat2quat(R).astype("float32")

                proj = (record["cam"] @ t.T).T
                proj = proj[:2] / proj[2]

                bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                bbox_obj = gt_info_dict[str_im_id][anno_i]["bbox_obj"]
                x1, y1, w, h = bbox_visib
                if self.filter_invalid:
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue

                mask_file = osp.join(
                    scene_root,
                    "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                )
                mask_visib_file = osp.join(
                    scene_root,
                    "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                )
                assert osp.exists(mask_file), mask_file
                assert osp.exists(mask_visib_file), mask_visib_file
                # load mask visib
                mask_single = mmcv.imread(mask_visib_file, "unchanged")
                mask_single = mask_single.astype("bool")
                area = mask_single.sum()
                if area < 3:  
                    self.num_instances_without_valid_segmentation += 1
                mask_rle = binary_mask_to_rle(mask_single, compressed=True)

                # load mask full
                mask_full = mmcv.imread(mask_file, "unchanged")
                mask_full = mask_full.astype("bool")
                mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)

                visib_fract = gt_info_dict[str_im_id][anno_i].get("visib_fract", 1.0)

                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox_visib,
                    "bbox_obj": bbox_obj,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": t,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                    "mask_full": mask_full_rle,
                    "visib_fract": visib_fract,
                    "xyz_path": None,  #  no need for test
                }

                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]
                insts.append(inst)
            if len(insts) == 0:  # filter im without anno
                continue
            record["annotations"] = insts
            dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "There are {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "There are {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.cache_dir, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.lmo_full.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3




def get_lmo_metadata(obj_names, ref_key):
    """task specific metadata."""

    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta



SPLITS_LMO = dict(
    lmo_bop_test=dict(
        name="lmo_bop_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        objs=ref.lmo_full.objects,
        ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test_targets_bop19.json"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="lmo_full",
    ),
)

for obj in ref.lmo_full.objects:
    for split in [
        "bop_test",
    ]:
        name = "lmo_{}_{}".format(obj, split)
        ann_files = [
            osp.join(
                DATASETS_ROOT,
                "BOP_DATASETS/lmo/image_set/{}_{}.txt".format(obj, split),
            )
        ]
        if name not in SPLITS_LMO:
            SPLITS_LMO[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                objs=[obj],  # only this obj
                scale_to_meter=0.001,
                ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test_targets_bop19.json"),
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=False,
                ref_key="lmo_full",
            )


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_LMO:
        used_cfg = SPLITS_LMO[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, LMO_BOP_TEST_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="lmo",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lmo_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_LMO.keys())



