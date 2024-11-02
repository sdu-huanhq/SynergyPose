
OUTPUT_DIR = "output/sp/lmo_pbr/SynergyPose"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=128,
    TOTAL_EPOCHS=60, 
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine", 
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.001),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("lmo_pbr_train",),
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=("datasets/BOP_DATASETS/lmo/test/test_bboxes/yolox.json",),
)

DATALOADER = dict(
    NUM_WORKERS=16,
    FILTER_VISIB_THR=0.3,
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP", 
    POSE_NET=dict(
        NAME="SynergyPose",
        XYZ_ONLINE=True,
        NUM_CLASSES=8,
        BACKBONE=dict(
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(3,),
                drop_path_rate=0
            ),
        ),
        CPD_HEAD=dict(
            INIT_CFG=dict(
                feat_dim=256,
                in_dim=1024,
                b1=0.4, 
                b2=0.6
            ),
            XYZ_CLASS_AWARE=True,
            MASK_CLASS_AWARE=True,
            OUT_LAYER_SHARED=False,
            DELTA_COOR=True,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu",type="ConvPnPNetNoRegion"),
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  
            XYZ_LOSS_MASK_GT="visib", 
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  
            MASK_LOSS_GT="trunc", 
            MASK_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=True, 
            PM_R_ONLY=True,  
            PM_LW=1.0,
            # CCLoss -----------
            DELTA_XYZ_LOSS_TYPE="L1",
            DELTA_XYZ_LW=5.0,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="lmo",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_bop19.json",
    ERROR_TYPES="mspd,mssd,vsd",
    RENDERER_TYPE="cpp", 
    SPLIT="test",
    SPLIT_TYPE="",
    N_TOP=1,  
    EVAL_CACHED=False,  
    SCORE_ONLY=False, 
    EVAL_PRINT_ONLY=False,  
    EVAL_PRECISION=False,  
    USE_BOP=True,  
)

TEST = dict(EVAL_PERIOD=1, VIS=False, TEST_BBOX_TYPE="est", USE_PNP=False)
