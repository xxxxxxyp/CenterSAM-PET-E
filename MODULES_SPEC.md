# Module Specs & I/O Contracts (New Prompt Generator Project)

本文件是本新项目的**实现规范与约束**（面向代码助手/实现者）。  
它与：
- `README.md`（项目边界与目录结构概览）
- `PLANS.md`（实践计划、闭环测试与验收）
协同工作。

> 本项目是全新工程，但为了兼容既有数据与运行习惯，**输入数据放置方式、部分字段命名与坐标约定沿用既有约定**。本文件将这些约定**完整写清楚**，实现时不得模糊引用外部项目。

---

## 0. Top-level Contract（总输入 / 总输出）

### 0.1 总输入（Input）
输入为 NIfTI 格式的 3D PET 影像：

- 文件格式：`.nii.gz`
- 数据类型：建议 `float32`（允许 float64）
- 通道：单通道（3D 体）

#### 轴顺序（强约束）
- **本项目全程使用 `XYZ` 轴顺序**
  - 内存数组：`volume_xyz[x, y, z]`
  - shape：`shape_xyz = [X, Y, Z]`
  - spacing：`spacing_xyz_mm = [sx, sy, sz]`
- **不允许对输入影像做轴转置**（no transpose / no reorientation）
- 只允许读取 NIfTI 并保留其数组顺序与 header 中的 spacing（zooms）

> 重要说明：mask 的使用以 **matrix 对齐（shape/索引对齐）**为准，不做物理空间对齐。你可以假设 mask 与 image 形状一致。

#### 输入目录结构（必须遵守）
```text
data/
└── processed/
    ├── images/
    │   ├── <case_id>.nii.gz
    │   └── ...
    ├── labels/                      # 仅训练/评估需要；推理可无
    │   ├── <case_id>.nii.gz
    │   └── ...
    └── body_masks/                  # 可选但推荐；与 images 形状一致
        ├── <case_id>.nii.gz
        └── ...
data/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

#### case_id 规则（必须）
- `case_id` 是 **4 位数字字符串**（例如 `"0042"`, `"1123"`）
- `splits/*.txt` 每行一个 `case_id`（不含扩展名）
- `domain` 由 `case_id` 第一位数字决定：
  - 第一位为 `0` → `FL`
  - 第一位为 `1` → `DLBCL`
  - 其他数字保留给未来 domain 扩展（必须允许但暂不启用）

> 约束：解析 splits 时必须验证 `case_id` 为 4 位数字；不符合则报错并给出行号。

---

### 0.2 总输出（Output）
每个病例输出一个 prompt 文件（默认 JSON）：

- 输出目录：`outputs/prompts/<case_id>/prompts_idS.json`

#### prompts.json 最小 schema（必须字段）
```json
{
  "case_id": "string",
  "domain": "FL",
  "image_path": "string",
  "shape_xyz": [X, Y, Z],
  "spacing_xyz_mm": [sx, sy, sz],
  "prompts": [
    {
      "prompt_id": "p000001",
      "box_xyz_vox": [x1, x2, y1, y2, z1, z2],
      "score": 0.0
    }
  ]
}
```

#### prompts.json 推荐扩展字段（建议但非必须）
```json
{
  "run": {
    "created_at": "2026-01-16T00:00:00Z",
    "config_path": "configs/prompt_pipeline.yaml",
    "config_hash": "sha256:...",
    "git_commit": "..."
  },
  "prompts": [
    {
      "source": {
        "center_xyz_vox": [xc, yc, zc],
        "heatmap_score": 0.92,
        "classifier_score": 0.81,
        "generator": "fixed_multiscale",
        "generator_params": {}
      }
    }
  ]
}
```

---

## 1. 坐标与边界约定（Critical Conventions）

### 1.1 轴顺序（全项目统一）
- 所有数组、坐标、shape、spacing 一律使用 **XYZ**
  - `volume.shape == (X, Y, Z)`
  - `spacing_xyz_mm == [sx, sy, sz]`

### 1.2 Box 坐标（必须统一）
`box_xyz_vox = [x1, x2, y1, y2, z1, z2]`

- **box 的 voxel index 指的是体素格点索引（array index）**，不是物理坐标。
- 半开区间：`[x1, x2) × [y1, y2) × [z1, z2)`
- 合法性：
  - `0 <= x1 < x2 <= X`
  - `0 <= y1 < y2 <= Y`
  - `0 <= z1 < z2 <= Z`
- 裁剪语义必须严格对应 Python slicing：
  - `crop = volume[x1:x2, y1:y2, z1:z2]`

所有模块输出 box 前必须：
- `clamp_to_image(box, shape_xyz)`，并再次校验合法性。

### 1.3 物理距离（mm）
若使用距离阈值/聚类（例如 eps_mm）：
- 必须按轴使用 `spacing_xyz_mm` 将 voxel 距离换算成 mm
- 禁止用固定 voxel 距离跨病例比较（spacing 可能不同）

### 1.4 确定性（Determinism）
相同输入、相同 config、相同模型权重下输出 prompts 必须确定一致。  
所有随机过程必须显式依赖 `seed`（在 config 中指定）。

---

## 2. Score 约定（Scoring Contract）

### 2.1 基本要求（必须）
- **同一病例内**：`score` 必须形成确定性的可排序全序（允许相等分数，但必须稳定 tie-break）
- **跨病例**：`score` 不要求可比（因为 K 自适应），但必须保留 `score` 供下游二次截断
- `score` 必须是 `float`
- `score` 不得为 `NaN/Inf`
- 推荐在写出前强制 `score = clamp(score, 0, 1)`（即使不是概率也允许映射）

### 2.2 排序要求（必须）
- 输出 `prompts` 数组必须按 `score` **降序**排列
- tie-break（必须固定一种）：
  1) `score` 降序
  2) `prompt_id` 升序（或 `box_xyz_vox` 字典序升序）

---

## 3. 模块划分与 I/O 合同（Module Contracts）

### 3.1 Module A: Case Index / Resolver
**职责**：由 `splits/*.txt` 与 `data/processed/*` 定位每个 case 的文件路径与 domain。

**输入**
- `data/splits/<split>.txt`
- `data/processed/images/<case_id>.nii.gz`
- 可选：`labels/<case_id>.nii.gz`、`body_masks/<case_id>.nii.gz`

**输出**
- `CaseRecord`：
  - `case_id`（4位数字字符串）
  - `domain`（由首位数字映射得到）
  - `image_path`
  - `label_path?`
  - `body_mask_path?`

**约束**
- 缺失文件必须报错并列出缺失清单
- 不允许 silent skip

---

### 3.2 Module B: NIfTI Loader（XYZ only）
**职责**：读取 `.nii.gz`，保持数组顺序为 XYZ，不做转置。

**输入**
- `image_path`
- `body_mask_path?`
- `label_path?`

**输出**
- `volume_xyz[X,Y,Z] float32`
- `meta`：
  - `shape_xyz`
  - `spacing_xyz_mm`（从 header zooms 直接取 `[sx, sy, sz]`）
  - `affine`（保留）
- `body_mask_xyz[X,Y,Z] bool`（若提供）
- `label_mask_xyz[X,Y,Z]`（若提供）

**约束**
- 只检查 shape 对齐：label/body_mask 必须与 image 的 `shape_xyz` 完全一致，否则报错
- 不做物理坐标对齐/重采样

---

### 3.3 Module C: Dummy Frontend（必须，用于闭环测试）
dummy 模式必须支持两种：

#### C1) dummy_from_label（仅训练/评估环境可用）
**输入**
- `label_mask_xyz`（GT）
**输出**
- `centers[]`：每个 GT lesion instance 产生一个 center
**预期**
- 在将 `K_cap_soft` 临时设为足够大（仅用于验证闭环）时，PromptRecall@K 接近 1.0，用于验证 box generator/selector/export/evaluate 的正确性。

#### C2) dummy_random（无 label 时可用）
**输入**
- `shape_xyz`、`gating_mask?`、`seed`
**输出**
- `centers[]`：随机点（数量由 config 指定）
**预期**
- pipeline 不 crash，prompts.json schema 正确（用于 smoke test）

---

### 3.4 Module D: Frontend Model（可替换）
**职责**：输出 heatmap 或 centers（如 CenterNet-PET）

**输入**
- `volume_xyz`
- `gating_mask?`

**输出**
- `heatmap_xyz[X,Y,Z]` 或 `centers[]`

**约束**
- 若输出 heatmap，v1.0 强制 `heatmap.shape == volume.shape`（避免坐标映射复杂度）

---

### 3.5 Module E: Peak Extractor（若前端输出 heatmap）
**职责**：确定性提取 Top-N peaks

**输入**
- `heatmap_xyz`
- `gating_mask?`
- `N_peaks`

**输出**
- `centers[]`（按 heatmap_score 降序）

**约束**
- ties 必须稳定排序（按坐标字典序）

---

### 3.6 Module F: Box Generator（v1.0 必须实现 fixed_multiscale）
**职责**：从 centers 生成候选 boxes（可多尺度）。

**输入**
- `centers[]`
- `shape_xyz`
- `spacing_xyz_mm`
- `generator = fixed_multiscale`
- `padding_ratio`（例如 0.2 表示边长放大 20%）

**输出**
- `proposals[]`：
  - `box_xyz_vox`
  - `center_xyz_vox`
  - `heatmap_score`
  - `generator="fixed_multiscale"`
  - `generator_params`

#### v1.0 推荐的多尺度集合（mm）
- `edge_mm_set = [12, 20, 32, 48]`（立方体边长，按 spacing 转 voxel）

**mm→voxel 取整规则（必须）**
- 对每个轴分别计算：`edge_vox_axis = ceil(edge_mm / spacing_axis_mm)`  
  （recall-first：宁可大一点避免截断）
- padding：`edge_vox_axis = ceil(edge_vox_axis * (1 + padding_ratio))`

---

### 3.7 Module G: Diversified Selection（v1.0 采用确定性网格聚类）
**职责**：多样性选择避免重复框挤占前列。

**v1.0 算法（必须）**
- 将 `center_mm = center_vox * spacing_xyz_mm`（逐轴）
- grid cell id：`cell = floor(center_mm / eps_mm)`（逐轴）
- 每个 cell 最多保留 `max_per_cell` 个 proposals（按 score）

**输入**
- `proposals_scored`
- `eps_mm`
- `max_per_cell`

**输出**
- `proposals_diverse`

---

### 3.8 Module H: GT lesion instance & center definition（用于评估与 dummy_from_label）
**连接性（必须统一）**
- 使用 3D **26-connectivity** 做连通域实例划分

**lesion center 定义（必须统一）**
- `center = round(mean(coords))`，其中 `coords` 为该实例所有体素坐标（x,y,z）
- round 规则固定为 `np.rint`（四舍五入到最近整数）

---

### 3.9 Module I: K Selector（Recall-first + soft cap）
**输入**
- `proposals_diverse`（按 score 排序）
- `K_min`（默认 20）
- `t_min`
- `K_cap_soft`（默认 200）

**输出**
- `prompts[]`

**约束**
- `K_out <= 200` 必须成立
- `K_out >= K_min`：若 proposals 不足则全部输出并记录 warning

---

### 3.10 Module J: Exporter
**职责**：写出 prompts.json 并校验 schema

**约束**
- 原子写入
- 写出后必须重新读取并通过 schema 校验（PLANS.md 反向测试）

---

## 4. 评估（Evaluation Contract）

- 评估在真实数据集上进行
- `evaluate_prompts.py` 输入：
  - `outputs/prompts/<case_id>/prompts.json`
  - `data/processed/labels/<case_id>.nii.gz`
- 输出 PromptRecall@K/FP@K（K=10/20/50/100/200）

> 注意：评估只用于 prompt 质量，不输出最终分割。

---

## 5. 禁止事项（Hard Constraints）

- 对输入影像做轴转置或物理方向重排（必须 XYZ only）
- 依赖 affine 做 mask 对齐（只按 shape/索引对齐）
- 产生 NaN/Inf score
- 输出 prompts 未按 score 排序
- 随机导致不可复现

---

## 6. 最小闭环（Minimal Closed Loop）

在 `frontend=dummy_from_label` 下：
- 在将 `K_cap_soft` 临时设为足够大（仅用于验证闭环）时，评估应得到 PromptRecall@K 接近 1.0

在 `frontend=dummy_random` 下：
- 必须输出 schema 正确的 prompts.json，且流程不 crash