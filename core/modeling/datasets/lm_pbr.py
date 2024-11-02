import hashlib
import logging
import os.path as osp
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint, lazy_property

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class LM_PBR_Dataset:
    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"] 

        self.dataset_root = data_cfg.get("dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr"))
        self.xyz_root = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop"))
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]
        self.scale_to_meter = data_cfg["scale_to_meter"] 

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  
        self.width = data_cfg["width"] 

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache")) 
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)


        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs] 
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  
        
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}  
        print("label2cat: {}".format(self.label2cat))
        
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs)) 
        
        print("obj2label: {}".format(self.obj2label)) 
        self.scenes = [f"{i:06d}" for i in range(50)]

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
        
        for scene in tqdm(self.scenes):
            scene_id = int(scene)
            scene_root = osp.join(self.dataset_root, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json")) 
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json")) 
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            
            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                rgb_path = osp.join(scene_root, "rgb/{:06d}.jpg").format(int_im_id)
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
                    "scene_im_id": scene_im_id,  
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "syn_pbr", 
                }
                insts = []
                
                
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids: 
                        continue
                    cur_label = self.cat2label[obj_id] 
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(pose[:3, :3]).astype("float32")
                    trans = pose[:3, 3]

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
                    if area < 32: 
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)


                    mask_full = mmcv.imread(mask_file, "unchanged")
                    mask_full = mask_full.astype("bool")
                    mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)

                    visib_fract = gt_info_dict[str_im_id][anno_i].get("visib_fract", 1.0)

                    xyz_path = osp.join(
                        self.xyz_root,
                        f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl",
                    )
                    
                    inst = {
                        "category_id": cur_label,  
                        "bbox": bbox_visib,
                        "bbox_obj": bbox_obj,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": trans,
                        "centroid_2d": proj,  
                        "segmentation": mask_rle,
                        "segmentation_path": mask_visib_file,
                        "mask_full": mask_full_rle,
                        "visib_fract": visib_fract,
                        "xyz_path": xyz_path,
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0: 
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
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
        models_info = mmcv.load(models_info_path) 
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs)))
        if osp.exists(cache_path) and self.use_cache:
            
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{ref.lm_full.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            
            
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height 




def get_lm_metadata(obj_names, ref_key):

    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {} 
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

LM_OCC_OBJECTS = [
    "ape",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
]
lmo_model_root = "BOP_DATASETS/lmo/models/"
################################################################################

SPLITS_LM_PBR = dict(
    lmo_pbr_train=dict(
        name="lmo_pbr_train",
        objs=LM_OCC_OBJECTS, 
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        scale_to_meter=0.001,
        with_masks=True,  
        with_depth=True, 
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="lmo_full",
    ),
)


# lmo single objs
for obj in ref.lmo_full.objects:
    for split in [
        "train",
    ]:
        name = "lmo_{}_{}_pbr".format(obj, split)
        if name not in SPLITS_LM_PBR:
            SPLITS_LM_PBR[name] = dict(
                name=name,
                objs=[obj], 
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr/xyz_crop"),
                scale_to_meter=0.001,
                with_masks=True, 
                with_depth=True, 
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=True,
                ref_key="lmo_full",
            )


def register_with_name_cfg(name, data_cfg=None):
    dprint("register dataset: {}".format(name))
    if name in SPLITS_LM_PBR:
        used_cfg = SPLITS_LM_PBR[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    
    DatasetCatalog.register(name, LM_PBR_Dataset(used_cfg))
    MetadataCatalog.get(name).set(
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lm_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_LM_PBR.keys())

