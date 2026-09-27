# -*- coding: utf-8 -*-
"""SBR 反应器阶段识别 —— 全局配置"""
import os

# ============ 路径 ============
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "videos")
YOLO_WEIGHTS = os.path.join(ROOT, "models", "best.pt")
FEATURE_DIR = os.path.join(ROOT, "outputs", "features")
CKPT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
RESULT_DIR = os.path.join(ROOT, "outputs", "results")

# ============ 阶段定义（按 SBR 工艺循环顺序） ============
# 数据集中“厌氧”即工艺讨论中的“缺氧”段（搅拌混合、无曝气）
CLASSES = ["进水", "进水后间歇", "缺氧", "好氧", "沉淀", "排水", "排水后间歇"]
NUM_CLASSES = len(CLASSES)

# 视频文件名 -> 类别索引
FILENAME_TO_CLASS = {
    "进水阶段": 0,
    "进水后间歇期": 1,
    "厌氧阶段1": 2,
    "厌氧阶段2": 2,
    "厌氧阶段3": 2,
    "好氧阶段1": 3,
    "沉淀阶段1": 4,
    "排水阶段": 5,
    "排水后间歇期": 6,
}

# 新增数据集（videos_new/）：帧匹配鉴定结果
NEW_FILENAME_TO_CLASS = {
    "进水阶段2": 0,        # 外部会话
    "厌氧阶段4": 2,        # 外部会话
    "好氧阶段2": 3,        # 外部会话
    "好氧阶段3": 3,        # 截取自测试视频
    "沉淀阶段2": 4,        # 截取自测试视频
    "排水阶段2": 5,        # 截取自测试视频
    "排水后间歇期2": 6,    # 外部会话
}
# 经帧匹配鉴定为“截取自测试视频”的片段（增量训练 v3 用，v2 需排除以防泄漏）
NEW_CLIPS_FROM_TEST = {"好氧阶段3", "沉淀阶段2", "排水阶段2"}
NEW_VIDEO_DIR = os.path.join(ROOT, "videos_new")

# ============ 全周期测试视频 ============
TEST_VIDEO = os.path.join(ROOT, "test_video", "full_cycle.mp4")  # 真值阶段区间（秒，左闭右开）
TEST_GT_INTERVALS = [
    (0,    275,  0),  # 0:00-4:35   进水
    (275,  530,  1),  # 4:35-8:50   进水后间歇
    (530,  1663, 2),  # 8:50-27:43  缺氧
    (1663, 4830, 3),  # 27:43-1:20:30 好氧
    (4830, 5293, 4),  # 1:20:30-1:28:13 沉淀
    (5293, 10**9, 5), # 1:28:13-结束  排水
]
# 机位分段与 ROI 裁剪框 (x1, y1, x2, y2)，基于 720x1280 原始帧
TEST_ROI_SEGMENTS = [
    (0,    1665, (40, 0, 700, 1280)),   # 广角：反应器主体
    (1665, 2960, (20, 30, 500, 950)),   # 特写：上部筒体+刻度尺
    (2960, 10**9, (40, 0, 700, 1280)),  # 广角
]

# ============ 特征提取 ============
SAMPLE_FPS = 2               # 每秒采样帧数（SBR 阶段转换为分钟级，2fps 足够）
CNN_INPUT_SIZE = 224         # EfficientNet-B0 输入尺寸
CNN_FEAT_DIM = 1280          # EfficientNet-B0 GAP 特征维度
DET_FEAT_DIM = 12            # YOLO 检测结构化特征维度
YOLO_CONF = 0.4              # 检测置信度阈值
EXTRACT_BATCH = 32           # 特征提取批大小

# YOLO 类别 ID（与 best.pt 训练时一致）
ID_MUD = 0      # 泥柱
ID_WATER = 1    # 水柱
ID_BUBBLE = 2   # 气泡

# ============ MS-TCN++ 模型 ============
TCN_HIDDEN = 256
TCN_STAGE1_LAYERS = 10       # 预测生成阶段 LID-TCN 层数
TCN_REFINE_LAYERS = 8        # 每个细化阶段 SS-TCN 层数
TCN_NUM_STAGES = 3           # 1 个生成阶段 + 2 个细化阶段
TCN_DROPOUT = 0.3

# ============ 训练 ============
CLIP_LENGTH = 128            # 训练片段长度（帧）= 64s @2fps
TRAIN_RATIO = 0.85           # 每个视频前 85% 用于训练，后 15% 验证
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_SMOOTH = 0.15         # 平滑损失权重
CLIPS_PER_VIDEO_PER_EPOCH = 24   # 每个视频每 epoch 采样片段数（平衡各阶段）
SEED = 42

# ============ 状态机 ============
SM_TRANSITION_BUFFER = 8     # 需连续 8 帧(4s)一致才允许状态转移

DEVICE = "cuda"


# ============ 新增测试视频（论文多视频验证） ============
# intervals: (起s, 止s, 类别id)，区间外帧标签为 -1（不计入指标）
TEST_VIDEOS2 = {
    "cycle5h_raw": {
        "file": r"path/to/VID_20260917_180113.mp4",
        "roi": (60, 0, 1040, 1900),  # 固定裁剪：反应器主体
        "intervals": [(0, 367, 0), (367, 638, 1), (638, 4285, 2),
                      (4285, 18690, 3), (18690, 18870, 4)],
    },
    "cycle5h_roi": {
        "file": os.path.join(ROOT, "test_video", "VID_20260917_180113_roi.mp4"),
        "intervals": [(0, 367, 0), (367, 638, 1), (638, 4285, 2),
                      (4285, 18690, 3), (18690, 18870, 4)],
    },
    "local_sep25": {
        "file": os.path.join(ROOT, "test_video", "2026_09_25_123546999.mp4"),
        "intervals": [(0, 360, 3), (360, 1667, 4)],
    },
    "local_sep21": {
        "file": os.path.join(ROOT, "test_video", "20260921_161425.mp4"),
        "intervals": [(0, 947, 3)],
    },
}
