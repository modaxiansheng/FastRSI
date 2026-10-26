# FastRSI 私有成果归档

归档日期：2026-10-26（北京时间）。这是筛选后的源码与成果备份，不是整个工作目录的镜像。

大文件、逐例结果和校验清单保存在私有数据仓库：
https://huggingface.co/datasets/pumpkin601/FastRSI-artifacts

## 内容

- `FastRSI/third_party`：已有本地核心依赖源码。
- `FastRSI/planning/libero_three_suites_20261024_success_v2`：成功评测使用的冻结源码、配置、资源覆盖层和场景资产。以其中的 `SOURCE_BINDINGS.json`、`RESOURCE05_BINDINGS.json`、`RESOURCE07_BINDINGS.json` 为版本依据。
- `BOUNDED_RECOVERY_SOURCE_SCOPE.json`：另外保留有正向结果的有限恢复实验所用冻结代码和最终凭据；保留开发集、预算及适用范围限制，不当作通用增益。
- `ORIGINAL_LIBERO_SNAPSHOT.json`：本次归档时已经闭合的逐例结果统计；不足500例的组仍为不完整快照。
- Hugging Face 中的报告包：最终报告、比较基线、原始统计和必要成本证据。

## 筛选规则

不保存废弃开发候选目录、探索阶段的整套源码快照、无用失败试验媒体、重复中间压缩包、环境缓存或公共大模型权重。冻结成功版本内部被引用的依赖模块随版本保留，不按文件名删除。正式评测的失败条目及成本证据保留，用于核对完整分母。成功画面和详细Agent轨迹按模型/套件/任务选取代表性样例。

原始工作目录与服务器实验记录没有被删除或修改。上传副本中的凭据会被移除；清单同时记录原始与上传字节的哈希。

## 解释边界

原版 LIBERO 与 LIBERO-Pro 分开；纯 VLA、PhyAgentOS P、FastRSI R 分开。未知结果不能算作失败。不同重试预算、不同版本的最高结果不能作为统一公平排名。`success_v2` 为有界恢复配置，不宣称跨回合学习。旧报告中的快照日期保留，不覆盖成今天的结果。

## 恢复

1. 克隆本私有仓库。
2. 登录 Hugging Face，下载私有 `pumpkin601/FastRSI-artifacts` 数据仓库。
3. 对照 `UPLOAD_MANIFEST.json` 校验包的 SHA256，再解压到新的工作目录；包内保留项目相对路径。
4. 公共模型按资产索引、模型配置和精确版本重新获取；不会把平台登录凭据放进归档。
5. 恢复后需要重新配置机器路径、容器镜像和运行环境。本次验证的是备份完整性，未重新执行GPU评测。

最终上传状态以 `UPLOAD_RECEIPT.json` 为准；仓库初始建立不代表所有包均已传完。
