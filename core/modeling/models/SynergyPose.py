import copy
import logging
import time
import timm
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage
from mmcv.utils import build_from_cfg
from mmcv.runner import load_checkpoint
from mmcv.runner.optimizer import OPTIMIZERS
from lib.torch_utils.solver.ranger import Ranger

from core.modeling.models.ClusterBranch import cluster_base

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .model_utils import (
    compute_mean_re_te,
    get_mask_prob,
    get_rot_mat,
    get_xyz_doublemask_out_dim,
)
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from core.utils.my_checkpoint import load_timm_pretrained
from CPDWithTokenMixerHead import CPDWithTokenMixerHead

logger = logging.getLogger(__name__)


class PoseEstimator(nn.Module):
    def __init__(
        self,
        nIn,
        featdim=128,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=True,
    ):
        super().__init__()
        self.featdim = featdim
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        conv_layer = StdConv2d if use_ws else nn.Conv2d
        self.features = nn.ModuleList()
        for i in range(num_stride2_layers):
            _in_channels = nIn if i == 0 else featdim
            self.features.append(
                conv_layer(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)
        for i in range(num_extra_layers):
            self.features.append(
                conv_layer(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(256, 3)


        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, coor_feat, extents=None, mask_attention=None):
        """
        Args:
            since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c in [3, 5] and self.denormalize_by_extent and extents is not None:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)
        # convs
        x = coor_feat

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        
        rot = self.fc_r(x)
        t = self.fc_t(x)
        return rot, t


class SynergyPose(nn.Module):
    def __init__(self, cfg, conv_branch, cpd_head=None, pnp_net=None, cluster_branch=None):
        super().__init__()
        self.conv_branch = conv_branch
        self.cluster_branch = cluster_branch
        
        self.CPDHead = cpd_head
        self.pnp_net = pnp_net

        self.cfg = cfg
        self.xyz_out_dim = 3
        self.mask_out_dim = 2
        

        if cfg.MODEL.POSE_NET.USE_MTL:
            self.loss_names = [
                "mask", "coor_x", "coor_y", "coor_z",
                "PM_R", "PM_T",
                "centroid", "z", "trans_xy", "trans_z", "rot",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )

    def forward(
        self,
        x,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_mask_full=None,
        gt_region=None, #!
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_coord_2d_rel=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        cpd_head_cfg = net_cfg.CPD_HEAD
        pnp_net_cfg = net_cfg.PNP_NET

        device = x.device
        bs = x.shape[0]
        num_classes = net_cfg.NUM_CLASSES
        out_res = net_cfg.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.conv_branch(x)  # [bs, c, 8, 8]
        coc_out = self.cluster_branch(x)
        
        vis_mask, full_mask, coor_x, coor_y, coor_z = self.CPDHead(conv_feat, coc_out)

        if cpd_head_cfg.XYZ_CLASS_AWARE:
            assert roi_classes is not None
            coor_x = coor_x.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if cpd_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            vis_mask = vis_mask.view(bs, num_classes, self.mask_out_dim // 2, out_res, out_res)
            vis_mask = vis_mask[torch.arange(bs).to(device), roi_classes]
            full_mask = full_mask.view(bs, num_classes, self.mask_out_dim // 2, out_res, out_res)
            full_mask = full_mask[torch.arange(bs).to(device), roi_classes]

        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if pnp_net_cfg.WITH_2D_COORD:
            if pnp_net_cfg.COORD_2D_TYPE == "rel":
                assert roi_coord_2d_rel is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d_rel], dim=1)
            else:  # default abs
                assert roi_coord_2d is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        mask_atten = None
        if pnp_net_cfg.MASK_ATTENTION != "none":
            mask_atten = get_mask_prob(vis_mask, mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE)

        pred_rot_, pred_t_ = self.pnp_net(coor_feat, extents=roi_extents, mask_attention=mask_atten)

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_cfg.Z_TYPE,
                is_train=do_loss, 
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in rot_type, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown trans type: {pnp_net_cfg.TRANS_TYPE}")

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TEST.USE_PNP or cfg.TEST.SAVE_RESULTS_ONLY or cfg.TEST.USE_DEPTH_REFINE:
                # TODO: move the pnp/ransac inside forward
                out_dict.update(
                    {
                        "mask": vis_mask,
                        "full_mask": full_mask,
                        "coor_x": coor_x,
                        "coor_y": coor_y,
                        "coor_z": coor_z,
                    }
                )
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
            }

            loss_dict = self.compute_loss(
                cfg=self.cfg,
                out_mask_vis=vis_mask,
                out_mask_full=full_mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                gt_mask_full=gt_mask_full,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
            )

            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def compute_loss(
        self,
        cfg,
        out_mask_vis,
        out_mask_full,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        gt_mask_full,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
    ):
        net_cfg = cfg.MODEL.POSE_NET
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj, "full": gt_mask_full}

        # xyz loss ----------------------------------
        gt_mask_xyz = gt_masks[loss_cfg.XYZ_LOSS_MASK_GT]
        loss_func = nn.L1Loss(reduction="sum")
        loss_dict["loss_coor_x"] = loss_func(
            out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        loss_dict["loss_coor_y"] = loss_func(
            out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        loss_dict["loss_coor_z"] = loss_func(
            out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
        ) / gt_mask_xyz.sum().float().clamp(min=1.0)
        
        loss_dict["loss_coor_x"] *= loss_cfg.XYZ_LW
        loss_dict["loss_coor_y"] *= loss_cfg.XYZ_LW
        loss_dict["loss_coor_z"] *= loss_cfg.XYZ_LW

        gt_mask = gt_masks[loss_cfg.MASK_LOSS_GT]
        loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask_vis[:, 0, :, :], gt_mask)
        loss_dict["loss_mask"] *= loss_cfg.MASK_LW


        # point matching loss ---------------
        assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
        loss_func = PyPMLoss(
            loss_type=loss_cfg.PM_LOSS_TYPE,
            beta=loss_cfg.PM_SMOOTH_L1_BETA,
            reduction="mean",
            loss_weight=loss_cfg.PM_LW,
            norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
            symmetric=loss_cfg.PM_LOSS_SYM,
            disentangle_t=loss_cfg.PM_DISENTANGLE_T,
            disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
            t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
            r_only=loss_cfg.PM_R_ONLY,
        )
        loss_pm_dict = loss_func(
            pred_rots=out_rot,
            gt_rots=gt_rot,
            points=gt_points,
            pred_transes=out_trans,
            gt_transes=gt_trans,
            extents=extents,
            sym_infos=sym_infos,
        )
        loss_dict.update(loss_pm_dict)


        if net_cfg.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict


def create_timm_model(**init_args):
    if init_args.get("checkpoint_path", "") != "" and init_args.get("features_only", True):
        init_args = copy.deepcopy(init_args)
        full_model_name = init_args["model_name"]
        modules = timm.models.list_modules()
        mod_len = 0
        for m in modules:
            if m in full_model_name:
                cur_mod_len = len(m)
                if cur_mod_len > mod_len:
                    mod = m
                    mod_len = cur_mod_len
        if mod_len >= 1:
            if hasattr(timm.models.__dict__[mod], "default_cfgs"):
                ckpt_path = init_args.pop("checkpoint_path")
                ckpt_url = pathlib.Path(ckpt_path).resolve().as_uri()
                timm.models.__dict__[mod].default_cfgs[full_model_name]["url"] = ckpt_url
                init_args["pretrained"] = True
        else:
            raise ValueError(f"model_name {full_model_name} has no module in timm")

    backbone = timm.create_model(**init_args)
    return backbone


def get_cpd_head(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    cpd_head_cfg = net_cfg.GEO_HEAD
    params_lr_list = []

    cpd_head_init_cfg = copy.deepcopy(cpd_head_cfg.INIT_CFG)

    xyz_num_classes = net_cfg.NUM_CLASSES if cpd_head_cfg.XYZ_CLASS_AWARE else 1
    mask_num_classes = net_cfg.NUM_CLASSES if cpd_head_cfg.MASK_CLASS_AWARE else 1
    
    cpd_head_init_cfg.update(
        xyz_num_classes=xyz_num_classes,
        mask_num_classes=mask_num_classes,
        xyz_out_dim=3,
        mask_out_dim=2,
        )
    
    cpd_head = CPDWithTokenMixerHead(**cpd_head_init_cfg)

    params_lr_list.append(
        {
            "params": filter(lambda p: p.requires_grad, cpd_head.parameters()),
            "lr": float(cfg.SOLVER.BASE_LR) * cpd_head_cfg.LR_MULT,
        }
    )

    return cpd_head, params_lr_list


def get_pose_estimatior(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    pnp_net_cfg = net_cfg.PNP_NET

    pnp_net_init_cfg = copy.deepcopy(pnp_net_cfg.INIT_CFG)

    pnp_net_init_cfg.update(
        nIn=5,
        rot_dim=6,
    )

    pnp_net = PoseEstimator(**pnp_net_init_cfg)

    params_lr_list = []
    params_lr_list.append(
        {
            "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
            "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
        }
    )
    return pnp_net, params_lr_list
    

def build_model_optimizer(cfg, is_test=False):
    net_cfg = cfg.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    params_lr_list = []
    
    # backbone
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    coc = cluster_base()
    params_lr_list.append(
            {"params": filter(lambda p: p.requires_grad, coc.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}
        )
    
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = create_timm_model(**init_backbone_args)

    params_lr_list.append(
        {"params": filter(lambda p: p.requires_grad, backbone.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}
    )
    
    cpd_head, cpd_head_params = get_cpd_head(cfg)
    params_lr_list.extend(cpd_head_params)

    pnp_net, pnp_net_params = get_pose_estimatior(cfg)
    params_lr_list.extend(pnp_net_params)

    # build model
    model = SynergyPose(cfg, conv_branch=backbone, cpd_head=cpd_head, pnp_net=pnp_net, coc=coc)
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    if is_test:
        optimizer = None
    else:
        OPTIMIZERS.register_module()(Ranger)
        optim_cfg = copy.deepcopy(cfg.SOLVER.OPTIMIZER_CFG)
        optim_cfg["params"] = params_lr_list
        optimizer = build_from_cfg(optim_cfg, OPTIMIZERS)
        # optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")

        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            logger.info("Check if the backbone has been initialized with its own method!")
            if backbone_pretrained == "timm":
                if init_backbone_args.pretrained and init_backbone_args.in_chans != 3:
                    load_timm_pretrained(
                        model.backbone, in_chans=init_backbone_args.in_chans, adapt_input_mode="custom", strict=False
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)
            logger.info(f"load backbone weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
