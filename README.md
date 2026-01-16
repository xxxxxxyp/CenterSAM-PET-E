# Prompt Generator for MedSAM2 (Pure PET, FL)

本仓库只实现 **MedSAM2 之前的部分**：从纯 PET（无 CT）影像中自动生成可直接输入 MedSAM2 的 **box prompts（3D bounding boxes）**。  
仓库不包含 MedSAM2 推理与最终精细分割。

- 输入：预处理后的 3D PET（可选 body mask）。
- 输出：每例一个 `prompts.json`（或等价格式），包含一组 3D boxes + scores + 必要元数据，供下游 MedSAM2 消费。

> 项目实施计划、闭环测试策略、里程碑与验收标准见 `PLANS.md`。


## Project Layout

```text
.
├── README.md
├── PLANS.md
├── configs/
│   ├── prompt_pipeline.yaml
|   ├── frontend_centernet.yaml
│   └── proposal_classifier.yaml
├── data/
│   ├── processed/
│   ├── splits/
│   └── cache/
├── models/
│   ├── frontend/
│   └── classifier/
├── outputs/
│   ├── prompts/
│   ├── proposals/
│   ├── metrics/
│   └── logs/
├── src/
│   └── promptgen/
│       ├── common/
│       ├── preprocessing/
│       ├── frontend/
│       ├── prompt/
│       ├── classifier/
│       └── pipeline/
└── scripts/
    ├── preprocess_dataset.py
    ├── train_frontend.py
    ├── train_classifier.py
    ├── generate_prompts.py
    └── evaluate_prompts.py
```


## Prompt Output Format (Interface)

每个病例输出一个 `prompts.json`（示例）：

```json
{
  "case_id": "XXXX",
  "spacing": [4.0, 4.0, 4.0],
  "prompts": [
    {
      "prompt_id": "p0001",
      "box_zyx_vox": [z1, z2, y1, y2, x1, x2],
      "score": 0.87,
      "source": {
        "center_zyx_vox": [zc, yc, xc],
        "heatmap_score": 0.92,
        "classifier_score": 0.81,
        "generator": "fixed_multiscale"
      }
    }
  ]
}
```

- `box_zyx_vox`：处理后 PET 空间的 voxel 坐标（坐标闭开区间约定在代码中固定，避免歧义）
- `score`：下游用于排序/截断
- `source`：调试与可解释性字段（建议保留）


## Scope (What we do / don't do)

We do:
- Prompt generation (3D box proposals) + scoring + export.

We don't:
- Run MedSAM2
- Produce final masks / Dice / TMTV

For implementation plan and testing strategy, see `PLANS.md`.