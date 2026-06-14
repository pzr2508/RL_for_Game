# RL for Game (Dueling DQN + Transformer)

基于强化学习的游戏自动控制框架。智能体通过观察屏幕像素并输出键盘/鼠标动作来学习玩游戏，使用 **Dueling DQN** 架构和 **Transformer** 时序编码。

---

## 功能特性

- **Dueling DQN** — 共享卷积骨干网络的 Value/Advantage 双分支架构
- **Transformer 时序编码器** — 基于自注意力机制的帧序列时序推理
- **多 GPU 训练** — 支持任意 GPU ID 的 DataParallel
- **Double DQN** — 可配置的目标网络估计，减少 Q 值过估计
- **动作均衡采样** — 基于逆频率加权采样，处理不均衡的动作分布
- **在线学习** — 实时经验回放，可选优先经验回放 (PER)
- **学习型奖励模型** — 基于 (state, action) 预测奖励的神经网络，用于在线学习
- **帧预处理** — 离线将 .jpg 转换为 .npy，消除训练 I/O 瓶颈
- **数据采集工具** — 屏幕录制 + Tkinter 标注界面，支持快捷键操作
- **屏幕捕获 + 推理循环** — 实时屏幕捕获、模型推理和游戏输入执行
- **热键 AI 开关** — F5 开启、F6 关闭 AI 智能体控制
- **TensorBoard 日志** — 训练/在线学习指标可视化

---

## 项目结构

```
rl_for_game/
├── main.py                          # 入口（train / inference / online / preprocess）
├── data_collection.py               # 屏幕录制 + 标注工具
├── data_loader.py                   # 数据集、CSV 缓存、奖励统计
├── online_train.py                  # 在线学习循环（推理 + 回放缓冲区训练）
├── pretrain_reward_model.py         # 从离线数据预训练奖励模型
├── preprocess_frames.py             # 将 .jpg 转换为 .npy 加快加载速度
├── resize_frames.py                 # 独立批量缩放工具
├── test.py                          # 简单 deque 行为测试（仅开发用）
├── config/
│   └── config.yaml                  # 统一 YAML 配置文件
├── core/
│   ├── bot_controller.py            # 热键监听 + AI 决策循环
│   ├── frame_buffer.py              # 线程安全单帧缓冲区
│   ├── recorder.py                  # 屏幕视频录制（XVID avi）
│   ├── screen_capture.py            # 后台屏幕捕获线程
│   └── vision_engine.py             # 帧预处理流水线
├── input/
│   ├── keyboard_controller.py       # 通过 pynput 的键盘输入
│   └── mouse_controller.py          # 通过 Win32 SendInput 的鼠标输入
├── logic/
│   └── decision.py                  # 动作 ID -> 键盘/鼠标执行映射
├── rl/
│   ├── agent.py                     # DuelingDQN 网络 + RLAgent
│   ├── trainer.py                   # 离线 Trainer + OnlineTrainer（含回放缓冲区）
│   ├── replay_buffer.py             # 统一回放缓冲区 + PER（SumTree）
│   └── reward_model.py              # 学习型奖励模型（CNN+LSTM+动作嵌入）
├── utils/
│   ├── config_loader.py             # YAML 加载器（深度合并 + 校验）
│   └── my_logger.py                 # 彩色控制台 + 每日轮换文件日志
├── models/                          # 保存的模型检查点
│   └── dueling_dqn_2000.pth
├── train_data/                      # 训练数据
│   ├── records.csv                  # 数据集索引
│   ├── online_transitions.csv
│   └── saved_videos/                # 按录制归类的 .jpg 帧目录
└── videos/                          # 录制的游戏视频（来自推理模式）
```

---

## 环境要求

Python 3.8+，需要以下核心包（详见 [requirements.txt](requirements.txt) 确切版本）：

| 包 | 用途 |
|---------|---------|
| torch / torchvision | Dueling DQN、Transformer、优化器 |
| opencv-python | 帧读写、缩放、letterbox |
| mss | 屏幕捕获 |
| pynput | 键盘模拟 |
| pyyaml | 配置加载 |
| tensorboard | 训练指标日志 |
| termcolor | 彩色控制台输出 |
| keyboard | 全局热键注册（Ctrl+Shift、F5/F6） |
| pydirectinput | 直接游戏输入（备选/预留） |

---

## 使用说明

### 1. 数据采集

```bash
python data_collection.py
```

以 `train_fps` FPS 录制屏幕。操作方式：
- **Ctrl+Shift** — 标记当前时刻，继续录制 N 秒，然后弹出标注对话框
- **标注对话框** — 选择执行的动作和观察到的奖励，然后按 **Ctrl+S** 保存

输出：按时间戳命名的帧目录（`train_data/saved_videos/`）+ `train_data/records.csv` 中的记录条目。

### 2. 帧预处理（推荐）

将所有 .jpg 帧转换为归一化的 .npy 文件，消除训练时的 I/O 瓶颈：

```bash
python main.py --mode preprocess
# 强制重新处理所有帧：
python main.py --mode preprocess --preprocess-force
# 控制并行工作线程数：
python main.py --mode preprocess --preprocess-workers 8
```

### 3. 离线训练

```bash
python main.py --mode train
# 自定义参数：
python main.py --mode train --batch-size 16 --epochs 3000 --gpus 0,1,2,3
```

每 `save_every` 轮保存一次检查点，训练结束后保存最终模型。

### 4. 推理（AI 玩游戏）

```bash
python main.py --mode inference
```

| 热键 | 功能 |
|--------|--------|
| **F5** | 启用 AI 智能体 |
| **F6** | 禁用 AI 智能体，重置内部状态 |

智能体实时捕获屏幕帧、运行模型推理，并执行键盘/鼠标动作。

### 5. 在线学习（实时训练）

```bash
python main.py --mode online
```

需要预训练的奖励模型（`models/reward_model.pth`）。智能体在玩游戏的同时，通过经验回放训练 DQN，并通过 TD-consistency 更新奖励模型。

### 6. 预训练奖励模型

```bash
python pretrain_reward_model.py
```
或
```bash
python pretrain_reward_model.py --epochs 1000 --batch-size 16 --gpus 0,1,2,3
```

在开始在线学习之前，使用离线 CSV 数据进行有监督预训练。

---

## 配置说明

所有配置集中在 [config/config.yaml](config/config.yaml)。命令行参数会覆盖对应的配置值。

| 配置段 | 关键参数 |
|---------|---------------|
| `app` | `recorder_fps`、`train_fps` |
| `screen` | `monitor_id`、`width`、`height` |
| `ai` | `continue_frames_num`（状态帧数）、`frams_resize` |
| `paths` | `model_path`、`records_csv`、`tensorboard_dir` |
| `train` | `batch_size`、`epochs`、`num_workers`、`balance_actions`、`gpu_ids` |
| `rl` | `gamma`、`lr`、`use_double_dqn`、`model_dim`、`transformer_layers/heads` |
| `inference` | `sleep_ms_when_empty`、`inference_topk`、`tie_delta` |
| `online` | `buffer_capacity`、`use_per`、`reward_model.*` |
| `record` | `enable`、`output_dir` |
| `log` | `level`、`file` |

---

## 模型架构

### DuelingDQN

```
输入：(frames, H, W) 灰度图序列
  │
  ├─ CNN 编码器：Conv2D(1->32->64->128) + BatchNorm + ResidualBlock
  ├─ 帧投影：Linear(128 -> model_dim) + LayerNorm
  ├─ 可学习位置编码
  ├─ TransformerEncoder（N 层，多头自注意力）
  │
  ├─ 时序注意力池化 ----+
  ├─ 最后一帧特征 ------+
  │                     │
  └─ 融合 -> Value 流 (1) + Advantage 流 (num_actions)
      │
      Q(s,a) = V(s) + A(s,a) - mean(A)
```

**核心技术：**
- **Dueling DQN**：分离 V 和 A 流，在动作无关的状态下提升动作评估质量。
- **Double DQN**：在线网络选动作，目标网络做评估——减少过估计偏差。
- **注意力池化**：所有时序特征的加权组合（不仅依赖最后一帧）。
- **平局打破**：推理时，若 top-2 Q 值差小于 `inference_tie_delta`，则在 top-K 动作内按 softmax 概率采样。

---

## 训练流程

1. **录制与标注** — 数据采集工具捕获屏幕帧并保存动作/奖励标注
2. **预处理** — 将 .jpg 转换为 .npy（灰度化 + letterbox + 归一化）
3. **离线训练** — 基于标注转换的有监督学习
   - 加载 CSV -> 构建帧序列 -> 动作均衡采样 -> Dueling DQN
   - 周期性地同步目标网络
4. **推理** — 加载已训练模型、捕获屏幕、执行动作
5. **在线学习**（可选）— 在推理过程中通过回放缓冲区和学习型奖励模型继续训练

---

## 关键设计决策

1. **奖励归一化** — 使用运行均值/标准差归一化（Welford 算法），确保跨不同奖励尺度的稳定训练。
2. **自动批量缩放** — 根据可用 GPU 显存相对于参考基线自动缩放批量大小。
3. **动作均衡** — 逆频率加权采样（`balance_alpha` 控制强度），防止多数动作偏置。
4. **优雅模型加载** — 即使网络架构变化也能加载兼容权重（部分加载并给出警告）。
5. **增量预处理** — 默认跳过已有的 .npy 文件；`--preprocess-force` 用于完全重新生成。
