# plan/ — OSCAR INT2 → vllm-ascend 0.23.0 适配方案索引

> 本目录是本工作区（new_oscar_vllmascend）**唯一**的前期方案/设计/路线 TODO 归属地。
> 规则：方案文档从 `design/PLAN-TEMPLATE.md` 复制标题矩阵，写作完成后在本索引登记一行，
> 并过 design gate（`bash <skill>/tools/gates.sh --level design --design-doc <file>`）。

## 索引

| 文件 | 一句话目标 | 状态 | 证据/门禁 |
|---|---|---|---|
| `PLAN-1-oscar-int2-vllm-ascend.md` | OSCAR INT2 KV 量化在 vllm-ascend 0.23.0 混合模型（Qwen3.5-27B-w8a8-mtp）上的零侵入插件方案 | DRAFT | 依据 references/oscar-vllm-pr46774 (57286d5d) + references/vllm (0fc695fc) + references/vllm-ascend (19e436985) |
| `ANALYSIS-20260904-A-vllmascend-kv-schedule.md` | vLLM-Ascend KV 调度全链路（spec→分组→16 池 P×nb→768/128 块表→三类执行→插件接线点），全部 file:line 证据 | 分析模式产物（纯文档） | 工作区 HEAD `1e95146` + references 树 |
| `ANALYSIS-20260904-B-oscar-mixed-precision.md` | Q1 混合精度能否释放显存（答：不能——池先定型+512×768 对齐等式与 160B 槽互斥）；Q2 混合机制落地状态与副作用（S1 每步全前缀反量化/27×、S2 ArgSort AiCpu、S3 +128MB 舞台、S5 pos2/3 残差） | 同上 | 同左 + 真机 05:20/05:21/05:35 日志 |

## 边界（plan/ vs docs/）

- `plan/`：一次性、还在论证中的方案/架构/TODO（本文档及其子目录）。
- `docs/`：（未来实现落地后）常青规范/契约/故障档案——本工作区尚未产生此类文档。
- 禁止把设计类 md 散落仓库根或其他目录。
