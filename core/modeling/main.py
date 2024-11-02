import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch

from loguru import logger as loguru_logger
import logging
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from setproctitle import setproctitle

from detectron2.data import MetadataCatalog
from mmcv import Config
import cv2
from pytorch_lightning import seed_everything
cv2.setNumThreads(0)  

cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer

from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str
import ref

from core.modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.modeling.engine.engine_utils import get_renderer
from core.modeling.engine.engine import SP_Lite
from core.modeling.models import SynergyPose

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True 

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = Config.fromfile(args.config_file) 
    if args.opts is not None:
        cfg.merge_from_dict(args.opts) 
        
    setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    bs_ref = cfg.SOLVER.get("REFERENCE_BS", cfg.SOLVER.IMS_PER_BATCH) 
    if bs_ref <= cfg.SOLVER.IMS_PER_BATCH:
        bs_ref = cfg.SOLVER.REFERENCE_BS = cfg.SOLVER.IMS_PER_BATCH
        accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1) 
    else:
        accumulate_iter = 1
    optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
    cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
    cfg.SOLVER.BASE_LR = optim_cfg["lr"]
    cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
    cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
    if accumulate_iter > 1:
        if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
            cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
                cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
            ) 
    if accumulate_iter > 1:
        cfg.SOLVER.WEIGHT_DECAY *= cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref

    register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        exp_id += "_test"
    cfg.EXP_ID = exp_id
    
    cfg.RESUME = args.resume
    return cfg


class Lite(SP_Lite):
    def set_my_env(self, args, cfg):
        my_default_setup(cfg, args) 
        seed_everything(int(cfg.SEED), workers=True)
        setup_for_distributed(is_master=self.is_global_zero)

    def run(self, args, cfg):
        self.set_my_env(args, cfg)
        train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        data_ref = ref.__dict__[train_dset_meta.ref_key]
        train_obj_names = train_dset_meta.objs
        render_gpu_id = self.local_rank
        logger.info("Render GPU ID: {}".format(render_gpu_id))
        renderer = get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=render_gpu_id)

        logger.info(f"Used module name: {cfg.MODEL.POSE_NET.NAME}")
        model, optimizer = SynergyPose.build_model_optimizer(cfg, is_test=args.eval_only)
        logger.info("Model:\n{}".format(model))

        if optimizer is not None:
            model, optimizer = self.setup(model, optimizer)
        else:
            model = self.setup(model)


        if args.eval_only: 
            MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return self.do_test(cfg, model)

        
        self.do_train(cfg, args, model, optimizer, renderer=renderer, resume=args.resume)
        return self.do_test(cfg, model)

@loguru_logger.catch
def main(args):
    cfg = setup(args)

    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")
    if args.num_gpus > 1 and args.strategy is None:
        args.strategy = "ddp"
    logger.info("using {} gpus".format(args.num_gpus))
    Lite(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.num_gpus,
        num_nodes=args.num_machines,
        precision=16 if cfg.SOLVER.AMP.ENABLED else 32, #! AMP: Automatic mixing precision
    ).run(args, cfg)


if __name__ == "__main__":
    parser = my_default_argument_parser()
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="the strategy for parallel training: dp | ddp | ddp_spawn | deepspeed | ddp_sharded",
    )
    args = parser.parse_args()
    iprint("Command Line Args: {}".format(args))

    main(args)
