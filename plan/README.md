# plan/ — OSCAR INT2 → vllm-ascend 0.23.0 适配方案索引

> 本目录是本工作区（new_oscar_vllmascend）**唯一**的前期方案/设计/路线 TODO 归属地。
> 规则：方案文档从 `design/PLAN-TEMPLATE.md` 复制标题矩阵，写作完成后在本索引登记一行，
> 并过 design gate（`bash <skill>/tools/gates.sh --level design --design-doc <file>`）。

## 索引

| 文件 | 一句话目标 | 状态 | 证据/门禁 |
|---|---|---|---|
| `PLAN-1-oscar-int2-vllm-ascend.md` | OSCAR INT2 KV 量化在 vllm-ascend 0.23.0 混合模型（Qwen3.5-27B-w8a8-mtp）上的零侵入插件方案 | DRAFT | 依据 references/oscar-vllm-pr46774 (57286d5d) + references/vllm (0fc695fc) + references/vllm-ascend (19e436985) |

## 边界（plan/ vs docs/）

- `plan/`：一次性、还在论证中的方案/架构/TODO（本文档及其子目录）。
- `docs/`：（未来实现落地后）常青规范/契约/故障档案——本工作区尚未产生此类文档。
- 禁止把设计类 md 散落仓库根或其他目录。
