# π₀.₅ 四种官方 Fine-tune Recipe：什么时候分别使用

## 已选定方案：Bench2Dex Task 26

后续实现统一使用以下不可变约定：

| 项目 | 选择 |
|---|---|
| Backbone / VLM | 冻结 SigLIP 与 Gemma VLM pipeline |
| 可训练主体 | 仅 action expert |
| Action projections | `action_in_proj` 与 `action_out_proj` 从零初始化并训练 |
| Action dimension | 58 |
| Action representation | 三选一：absolute、`a[t+k]-s[t]`、逐时间点对齐的 `a[t+k]-s[t+k]`（全部 58 维） |
| Main camera | `cam_overhead` → `base_0_rgb` |
| Left wrist camera | `cam_wrist_left` → `left_wrist_0_rgb` |
| Right wrist camera | `cam_wrist_right` → `right_wrist_0_rgb` |
| Dataset | Task 26 `replay-generalization` 原生 HDF5 lazy loading |
| Checkpoints | `/mnt/ssd1/szhao/DexTool/openpi_checkpoints` |
| Logs / WandB | `/mnt/ssd1/szhao/DexTool/openpi_logs` |
| Assets / cache | `/mnt/ssd1/szhao/DexTool/openpi_assets` |
| Virtual environment | `/mnt/ssd1/szhao/DexTool/openpi_env` |

数据管线已使用 Task 26 的 `episode_000000.hdf5` 端到端验证：homing 截断、有效帧过滤、三路 JPEG 解码、58 维 state/action、quantile normalization、224×224 resize、prompt tokenization 和 batching 均通过。完整 100 条数据也已通过 schema 扫描；`action_horizon=50` 时共有 59,038 个可训练 frame-start samples。

三种 action label 通过独立 config 切换：

| 模式 | Config | Label | 推理时恢复 absolute target |
|---|---|---|---|
| Absolute | `pi05_bench2dex_task26_action_expert` | `a[t+k]` | 不转换 |
| Chunk-start state delta | `pi05_bench2dex_task26_action_expert_state_delta` | `a[t+k]-s[t]` | `s[t]+delta[k]` |
| Time-aligned dynamics delta | `pi05_bench2dex_task26_action_expert_aligned_delta` | `a[t+k]-s[t+k]` | 每步执行时用实时 `s[t+k]+delta[k]` |

三种模式的 `action_horizon` 都保持 50；aligned delta 不把 horizon 设为 1。
模型仍一次预测 50 个 delta，但 evaluator 在消费 action queue 时逐步读取
实时 qpos，并计算 `absolute_target=s_live+delta`，不累计之前的 tracking
error。三种 config 必须分别计算 normalization statistics。旧的 absolute
config 和 checkpoint 语义保持不变。

> 注意：发布的 Task 26 全部 100 个 replay 文件均标记为 `meta/success=False` 和 `meta/demo_eligible=False`。当前 loader 与 Bench2Dex 官方 π₀.₅ loader 一样不按这两个字段过滤；在正式训练前需要单独决定是否接受这些 demonstrations，或向数据发布方确认 metadata 的含义。

---

你问的是这 **4 套官方 recipe 分别适合什么情况**：

1. `pi05_libero`
2. `pi05_aloha_pen_uncap`
3. `pi05_droid_finetune`
4. `pi05_full_droid_finetune`

它们最大的区别其实不只是超参，而是 **“你从哪个 checkpoint 开始 + 你的数据规模 + embodiment 是否和预训练/中间 finetune 一致”**。

| 配置 | 从哪里开始 | 数据规模 | 适合什么时候 |
|---|---|---:|---|
| `pi05_libero` | `pi05_base` | 中等/较大 | **新的 robot/task，和已有 PI embodiment 没有特别强对应关系** |
| `pi05_aloha_pen_uncap` | `pi05_base` | 小数据 | **ALOHA / 双臂 / task-specific，小规模 demos** |
| `pi05_droid_finetune` | **`pi05_droid`** | 小数据，官方写 `< tens of hours` | **你的 robot/data 很像 DROID / Franka setup** |
| `pi05_full_droid_finetune` | `pi05_base` | **整个 DROID 级别** | **做一个新的 generalist robot foundation model / 大规模 domain adaptation** |

---

## 1. `pi05_libero`

官方参数大概是：

```text
checkpoint = pi05_base

batch = 256
LR = 5e-5
warmup = 10k
steps = 30k
EMA = 0.999
action_horizon = 10
```

### 我什么时候推荐它？

它最适合这种情况：

> **“我有一个自己的 robot dataset，而且我的 embodiment / task 并没有一个 PI 官方已经专门训练好的 checkpoint 可以继承。”**

比如你有：

- FANUC + gripper
- UR5 + 自己的 gripper
- Kinova
- 自己的 custom arm
- 非 DROID-compatible Franka setup
- 自己定义的 observation/action space

这时候：

\[
\pi_{0.5}\text{-base}
\rightarrow
\text{your dataset}
\]

是最自然的。

### 为什么它这么 aggressive？

因为 LIBERO 本身并不是“100 条完全一样的单任务 demo”。

它包含多个 manipulation task，数据 distribution 更宽。

因此官方用了：

```text
batch 256
LR 5e-5
```

本质是在说：

> 我有足够多、足够 heterogeneous 的数据，可以允许 base model 做比较大的 adaptation。

所以我会把这个 recipe 理解成：

### **Generic full-finetuning recipe**

而不是：

> 所有自己的 dataset 都应该 batch 256 + 5e-5。

### 适合

例如：

```text
1000+ demos
multiple tasks
multiple objects
scene variation 很大
```

或者你真的是想做：

> π₀.₅ → 新 embodiment adaptation

我会比较倾向 LIBERO recipe。

### 不太适合

你现在：

```text
~100 trajectories
single/few tasks
```

直接：

```text
batch 256
LR 5e-5
30k
```

我觉得偏 aggressive。

---

## 2. `pi05_aloha_pen_uncap`

这个跟你现在的数据规模**最像**。

官方：

```text
checkpoint = pi05_base

batch = 64
steps = 20k
```

其他 optimizer 使用默认设置。

这是一个非常典型的：

> **small task-specific finetune**

它解决的是：

```text
已经有 π0.5 base
        ↓
大概百条左右自己的 demonstrations
        ↓
让它学会一个具体 manipulation task
```

### 我什么时候推荐这个？

如果你：

- 大约 **50–300 trajectories**
- task 数量很少
- 每个 task 相对明确
- 数据都是你自己 teleop
- 不打算做一个“大一统 generalist”
- 只是希望 π₀.₅ 学会你的 task

那么这套通常就是第一选择。

比如：

### Single arm

```text
pick cup
put cup on rack
open drawer
insert object
pick-and-place
```

或者 dual-arm：

```text
fold
uncap
handover
open container
```

都可以借鉴这套 **training intensity**。

### 注意：不要求你真的用 ALOHA

这一点很重要。

不要理解成：

> 我不是 ALOHA，所以不能用这个 recipe。

不是。

它更大的意义是给你一个：

### **small-data task-specific π₀.₅ recipe**

也就是：

```text
π05_base
+
~100 demos
+
moderate batch
+
moderate LR
+
20k training
```

---

## 3. `pi05_droid_finetune`

这个和前两个有一个**本质区别**。

它不是：

\[
\pi_{0.5}\text{-base}
\rightarrow
\text{your task}
\]

而是：

\[
\boxed{
\pi_{0.5}\text{-DROID}
\rightarrow
\text{your task}
}
\]

官方明确加载的是：

```text
gs://openpi-assets/checkpoints/pi05_droid/params
```

而且明确写着：

> fine-tuning pi05-DROID on a custom (smaller) DROID dataset.

官方还特别说，这里的 custom DROID dataset 可以是 **小于几十小时**，并建议转换成 LeRobot format。

参数是：

```text
batch = 32
steps = 20k

action_horizon = 16
action_dim = 32
```

### 这个什么时候最值得用？

如果你自己的 setup 和 DROID **非常接近**。

比如：

```text
Franka
+
gripper
+
EEF Cartesian control
+
DROID-like cameras
+
tabletop manipulation
```

这时候：

```text
pi05_droid
```

可能已经对你的：

- embodiment
- action distribution
- camera geometry
- manipulation style

有非常好的 prior。

所以你只需要：

> teach 它你的 particular task。

这就相当于：

\[
\text{foundation model}
\rightarrow
\text{robot-specialized model}
\rightarrow
\text{task-specialized model}
\]

比：

\[
\text{foundation model}
\rightarrow
\text{task}
\]

更容易。

### 为什么 batch 反而只有 32？

因为这里不是在做大的 representation/domain adaptation。

模型已经经过：

```text
π05 base
      ↓
full DROID
      ↓
π05-DROID
```

你现在只是最后：

```text
π05-DROID
      ↓
my task
```

所以需要的 adaptation 很小。

我会把它理解为：

### **specialist → specialist finetuning**

因此官方很保守：

```text
batch 32
20k
```

而且最重要的是：

> 官方建议继续使用原始 DROID normalization statistics。

### 什么时候不应该用 `pi05_droid_finetune`？

比如你是：

```text
FANUC
joint-space action
完全不同 camera
完全不同 gripper
```

那我不会因为：

> “DROID checkpoint 更强”

就硬上 `pi05_droid`。

因为此时 DROID-specific prior 反而可能是 constraint。

这种情况下：

```text
pi05_base
```

可能更干净。

---

## 4. `pi05_full_droid_finetune`

这个和你现在的情况**基本不是一类问题**。

参数：

```text
checkpoint = pi05_base

batch = 256
LR = 5e-5
warmup = 1k
steps = 100k
```

而且官方注释直接说：

> fine-tuning pi05 on the **full DROID dataset**

并专门使用 RLDS loader，因为完整 DROID 数据量很大。

### 它实际是在做什么？

更像：

\[
\pi_{0.5}\text{-base}
+
\text{huge robot-specific dataset}
\]

得到：

\[
\boxed{
\pi_{0.5}\text{-DROID generalist}
}
\]

不是在：

> 学一个 task。

而是在：

> **把 π₀.₅ 整体适配到一个 robot ecosystem。**

比如它学：

```text
pick
place
drawer
cabinet
object manipulation
kitchen
table
不同 objects
不同 scenes
...
```

最终变成一个 Franka generalist。

---

# 所以这四个其实可以画成这样

## Path A：自己的新 robot

```text
                  your medium/large dataset
π0.5 base  ────────────────────────────────> your model
                     ↑
                 LIBERO recipe
```

## Path B：自己的小 task

```text
                   ~100 demos
π0.5 base  ────────────────────────────────> task model
                     ↑
                  ALOHA recipe
```

## Path C：DROID-like robot + 小 task

```text
π0.5 base
     │
     │ huge DROID
     ↓
π0.5-DROID
     │
     │ your ~100 demos
     ↓
your task
     ↑
DROID finetune recipe
```

## Path D：自己也有巨量 robot data

```text
π0.5 base
     │
     │ 10^5~10^6+ diverse robot samples
     │ many tasks
     │ many scenes
     ↓
robot-specific generalist
     ↑
full-DROID style recipe
```

---

# 最重要的选择标准其实只有两个

## Q1：我有没有一个“和我的 robot 特别匹配”的 pretrained specialization？

如果有，例如：

```text
你的 setup ≈ DROID
```

那么：

> **优先 `pi05_droid_finetune`**

因为它已经有 embodiment prior。

如果没有：

> **从 `pi05_base` 开始。**

---

## Q2：我的数据是 task-specific 小数据，还是 generalist 大数据？

### 小数据

比如：

```text
50–300 demos
1–3 tasks
```

→

> **ALOHA-style**

```text
batch ~64
20k
默认较保守 LR
```

### 中等 / heterogeneous

比如：

```text
很多 tasks
几百到几千 demos
object / scene variation 很大
```

→

> **LIBERO-style**

```text
batch 128–256
LR ~5e-5
30k+
```

### 巨大 generalist dataset

比如：

```text
数十/数百小时
many robots / scenes / tasks
```

→

> **full-DROID-style**

```text
batch 256
5e-5
100k
```

---

# 对你现在 100 条数据，我会怎么选

如果你现在是：

> **普通 single arm + gripper，而且不是标准 DROID/Franka setup**

我的顺序非常明确：

### 第一选择

```text
pi05_aloha_pen_uncap-style
```

也就是：

```text
π05_base
batch = 64
20k
```

它最接近你现在这个：

> **foundation model + ~100 task-specific demos**

的问题。

---

如果你的 robot 恰好是：

> **Franka + gripper，而且 action/camera/data format 和 DROID 很接近**

那我反而会第一优先：

```text
pi05_droid_finetune
```

即：

```text
π05-DROID
batch = 32
20k
```

因为这相当于已经先替你做过一次 embodiment adaptation。

---

而我目前**不会选**：

```text
pi05_libero
```

除非你的 100 条其实非常 heterogeneous，比如每条里面覆盖很多不同 task/object/scene；更不会选：

```text
pi05_full_droid_finetune
```

因为 100 demos 和 full-DROID 这种训练 regime 完全不是一个数据量级。

所以如果让我把它们按你的情况排序：

\[
\boxed{
\text{ALOHA-style}
>
\text{DROID-style（如果 embodiment match）}
\gg
\text{LIBERO-style}
\gg
\text{full-DROID-style}
}
\]

这里面最关键的不是“ALOHA 和你的机器人像不像”，而是 **ALOHA recipe 的数据规模和 optimization regime 最像你现在的 100-demo task-specific finetuning**。



| Setting | Full finetune | Freeze backbone，只训 action expert |
|---|---:|---:|
| Trainable | 全部 π₀.₅ | action expert + action in/out projection |
| Batch size | **32–64** | **64** |
| Peak LR | **2.5e-5** | **5e-5** |
| Warmup | **1k steps** | **1k steps** |
| Train steps | **8k–15k** | **15k–20k** |
| EMA | **0.99** | **0.99** |
| Grad clip | **1.0** | **1.0** |
| Weight decay | 基本 0 / 官方默认 | 基本 0 / 官方默认 |
| Precision | bf16 | bf16 |
| Overfit 风险 | **较高** | **较低** |
| Adapt 新视觉域能力 | **更强** | 较弱 |
| 我对 100 条数据的偏好 | 第二选择 | **第一选择** |
