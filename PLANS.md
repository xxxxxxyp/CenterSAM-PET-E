# Project Practice Plan (with Closed-Loop Forward Implementation + Reverse Testing)

本文件定义本项目的实践规划：每一个增量构建都必须具备**正向可运行**与**反向可验证**（测试闭环），确保每一步输出与预期一致，避免后期“系统能跑但不知道哪里错”的情况。

> 关键词：闭环（Closed Loop）= 代码实现 + 可重复测试 + 指标/断言 + 可视化/日志 + 失败可定位


---

## 0. 总原则（Non-negotiables）

1) **先写测试，再写实现（Test-first for contracts）**  
   对“接口协议（prompts.json）”“坐标与IoU”“匹配规则”“排序/截断”等关键逻辑，必须先有单元测试或黄金样例（golden files）。

2) **每个模块都有可验证的输入输出（Contract）**  
   - 输入是什么（shape / dtype / coordinate convention）
   - 输出是什么（schema / invariants）
   - 不允许“靠肉眼看对不对”

3) **每一步都有反向测试（Reverse Test / Consistency Check）**  
   例如：
   - box → mask → bbox 的可逆性检查
   - 坐标映射前后的一致性（映射回去应接近原值）
   - 排序/截断的单调性与稳定性（相同输入结果必须确定性一致）

4) **每次合并/里程碑必须具备端到端最小闭环（E2E smoke）**  
   在一个极小样本集（例如 2-5 例）上：
   - 能生成 prompts.json
   - 能跑 evaluate_prompts 得到指标
   - 输出日志/中间产物可追溯


---

## 1. 目标交付物（Deliverables）

- D1：`prompts.json` 规范稳定（schema + 坐标约定 + 单测）
- D2：`evaluate_prompts.py` 可计算 PromptRecall@K / FP@K（含 per-case 明细）
- D3：最小 pipeline：`generate_prompts.py` 在小样本上可稳定输出（无需训练模型也能跑：支持 dummy 前端）
- D4：前端候选中心（CenterNet 或替代）输出可插拔
- D5：box 生成/多样性选择/截断策略完整闭环 + 测试
- D6：proposal re-ranker（分类器）可选接入 + 测试
- D7：持续集成（本地或CI）能自动跑：单测 + 小样本E2E


---

## 2. 测试分层策略（Testing Pyramid）

### 2.1 单元测试（Unit）
关注“纯函数”和“契约”：
- bbox IoU / intersection / volume
- box clamp 到 image 边界
- center-in-box 判定
- 3D peak 提取（给定 toy heatmap 的确定性输出）
- diversify（cluster-aware selection 的 determinism）
- K 选择逻辑（K_min / t_min / K_cap_soft 的边界情况）
- prompts.json schema 验证（字段必备、类型正确、坐标合法）

### 2.2 集成测试（Integration）
- 以一个 toy volume + toy GT lesions 构造完整链路：
  - centers → boxes → rank → select → export → evaluate
- 使用 “golden prompts.json” 对比，确保输出确定性（或在容许范围内一致）

### 2.3 端到端烟雾测试（E2E Smoke）
- 在极小样本数据集（2-5例）上跑：
  - `scripts/generate_prompts.py`
  - `scripts/evaluate_prompts.py`
- 断言输出文件存在、schema正确、指标不为 NaN、耗时不超阈值

> 注：E2E 不追求性能好，只追求“闭环跑通、结果可解释、可重复”。


---

## 3. 里程碑与闭环清单（Milestones with Forward + Reverse Loops）

下面每个阶段都包含：
- ✅ 正向实现（Forward Implementation）
- 🔁 反向测试（Reverse Tests）
- 📦 产出物（Artifacts）
- 🧪 验收（Acceptance Criteria）

---

### M0：定义接口与评估口径（先定规则）
✅ Forward
- 定义 prompts.json schema（字段、坐标顺序、闭开区间约定）
- 定义匹配规则：center-in-box 为主 + IoU≥0.1 为辅
- 定义一对一匹配策略（GT 为主，按 score 优先）

🔁 Reverse
- schema validator：缺字段/类型错误必须报错
- 坐标合法性断言：z1<z2, 坐标在图像范围内（或 clamp 后仍合法）

📦 Artifacts
- `src/promptgen/pipeline/formats.py`（schema / dataclass）
- `tests/test_formats.py`

🧪 Acceptance
- 运行 `pytest` 通过
- 任意 prompts.json 都能被 validator 判定为 valid/invalid（不允许 silent fail）

---

### M1：几何与指标引擎（metrics first）
✅ Forward
- 实现：
  - bbox IoU（3D）
  - center-in-box
  - GT lesion 连通域提取（从 GT mask）
  - PromptRecall@K、FP@K、per-case 明细输出

🔁 Reverse
- 对 toy case（手工构造GT与boxes）写“真值断言”：
  - IoU 结果精确等于预期
  - Recall@K 结果精确等于预期
  - 一对一匹配不会重复计数

📦 Artifacts
- `src/promptgen/common/geometry.py`
- `src/promptgen/common/metrics.py`
- `tests/test_geometry.py`
- `tests/test_metrics.py`

🧪 Acceptance
- 关键函数覆盖率达到基本水平（建议 >80% for common/）
- metrics 输出与 toy 真值完全一致

---

### M2：最小 pipeline（不依赖训练，也能闭环）
✅ Forward
- 实现 `generate_prompts.py` 支持 **dummy 前端**：
  - 直接从 GT lesion center（或随机）生成 centers
  - 走完整 pipeline：box_generator → select_k → export prompts.json

🔁 Reverse
- E2E smoke（2-5例）：
  - 生成 prompts.json
  - evaluate_prompts 输出 Recall@K 合理（dummy 如果用 GT center 应接近 1.0）
- 结果可重复（固定 seed）

📦 Artifacts
- `scripts/generate_prompts.py`
- `scripts/evaluate_prompts.py`
- `tests/test_e2e_smoke.py`（小样本）

🧪 Acceptance
- 不训练任何模型，pipeline 也能跑通并产生可评估 prompts
- 任何人 clone 后可复现 smoke test

---

### M3：peak 提取与 box 生成策略（真实前端的前置工程）
✅ Forward
- 实现 3D peak 提取（maxpool NMS）
- 实现固定多尺度 box 生成 + padding
- 引入 body mask gating（mask 外 heatmap 置零或不搜索）

🔁 Reverse
- peak 提取 toy 测试：输入特定 heatmap，应输出指定 peaks
- box 生成测试：
  - 对给定 center、给定 spacing/patch size，输出 box 与预期一致
  - clamp 后仍合法

📦 Artifacts
- `src/promptgen/frontend/peaks.py`
- `src/promptgen/prompt/box_generator.py`
- `tests/test_peaks.py`
- `tests/test_box_generator.py`

🧪 Acceptance
- peak 提取确定性、对边界情况无异常
- box 生成不会产生越界/无效 box

---

### M4：前端模型接入（CenterNet 可替换）
✅ Forward
- 接入前端模型推理（CenterNet heatmap）
- Pipeline 支持 `--frontend {dummy, centernet}` 切换

🔁 Reverse
- regression test：固定一例输入与固定模型权重，输出 prompts.json hash/关键字段稳定
- failure test：模型文件不存在、shape 不符时给出清晰错误

📦 Artifacts
- `src/promptgen/frontend/infer.py`
- `scripts/train_frontend.py`（可后置，只要 infer 可用）

🧪 Acceptance
- 能在小样本上生成 prompts.json（即使指标一般，也要闭环可跑）
- 失败时可定位（日志含 case_id、阶段、异常类型）

---

### M5：排序/多样性/自适应K（Recall-first 版本）
✅ Forward
- 实现 diversify（cluster-aware selection）
- 实现 K 选择：
  - `K_min`
  - `t_min`
  - `K_cap_soft=200`

🔁 Reverse
- determinism test：同输入同输出（排序、聚类、截断不能随机）
- boundary tests：全部分数很低/很高、候选为0、候选>200等

📦 Artifacts
- `src/promptgen/prompt/diversify.py`
- `src/promptgen/prompt/select_k.py`
- `tests/test_select_k.py`
- `tests/test_diversify.py`

🧪 Acceptance
- K 选择逻辑严格符合规格
- 不会出现“输出0个prompt”的意外（除非明确允许）

---

### M6：proposal re-ranker（可选，但能显著提高“真阳性靠前”）
✅ Forward
- 接入分类器推理，生成 `classifier_score`
- score fusion（如 0.7 cls + 0.3 heatmap）

🔁 Reverse
- stub classifier：固定输入输出固定分数，验证融合逻辑正确
- e2e：加入 classifier 后 prompts 排序发生可预期变化

📦 Artifacts
- `src/promptgen/classifier/infer.py`
- `tests/test_score_fusion.py`

🧪 Acceptance
- classifier 可选开关不影响 pipeline 稳定性
- 关闭 classifier 时退化为 heatmap-only 版本

---

## 4. 数据与实验闭环（Data Loop）

- 建立一个最小 **devset**（建议 5-20 例）：
  - 包含低负荷（1-2灶）
  - 包含高负荷（腿部多发）
  - 包含易混淆器官热点病例
- 所有 smoke / 回归测试都只在 devset 上跑，确保快速。

输出必须包括：
- per-case prompts 数量
- per-case Recall@{20,50,200}
- per-case FP@{20,50,200}
- 失败案例列表（按 case_id）


---

## 5. 日志规范（Logging Spec）

每个 case 至少记录：
- `case_id`
- `num_peaks`
- `num_boxes_generated`
- `num_boxes_after_diversify`
- `K_out`（最终输出 prompts 数）
- top-10 分数摘要（heatmap/cls/final）
- 运行耗时分解（peak / box / rank / export）


---

## 6. 风险清单与防呆（Risk & Safeguards）

- 坐标系错乱：必须有“映射一致性测试”与“可视化抽查脚本”（保存 overlay）
- 输出不可控：K_cap_soft=200 作为熔断；超出要报警并保存中间产物
- 结果不可复现：任何随机过程必须固定 seed，排序必须稳定
- 指标漂移：任何口径变更必须更新 tests 与 golden files


---

## 7. 你接下来可以怎么用这份计划

1) 先实现 M0-M2（不训练也能闭环）
2) 再实现 M3-M5（把 prompt 生成工程做扎实）
3) 最后接入模型（M4）与 re-ranker（M6）

这样每一步都有明确的“正向能跑 + 反向能测”的闭环，不会陷入“跑了半个月不知道哪里出了问题”的状态。