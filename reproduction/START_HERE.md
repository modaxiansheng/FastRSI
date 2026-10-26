# FastRSI 换设备恢复与续跑记录

记录日期：2026-10-27，北京时间。数据快照取自 2026-10-26，云端文件复核时间见 `BACKUP_RECORD.json`。本记录覆盖已筛选的最终有效版本，不是服务器整盘镜像。

## 1. 先看这些文件

| 文件 | 用途 |
|---|---|
| `BACKUP_RECORD.json` | 三个仓库的固定提交、六组模型、五个压缩包、文件大小、SHA256 与备份边界 |
| `PATH_MAP.json` | 原服务器路径 → 新工作目录相对路径；仅映射，不自动改动冻结代码 |
| `ENVIRONMENT_PACKAGES.json` | 从归档读取的 170 条软件包元数据及三个容器镜像身份；不是完整环境锁文件 |
| `RESUME_STATE.json` | 已上传实验快照的进度和未完成部分 |
| `COMPLETED_CASE_INDEX.jsonl` | 5,968 个已完成案例的 campaign、任务、初始状态、记录路径和 SHA256 |
| `PENDING_CASE_INDEX.jsonl` | 快照中尚缺的 32 个案例身份；仅用于核对，不自动运行 |
| `verify_restore.py` | 统一下载校验与安全解压工具，不启动模型、模拟器或训练 |
| `NEW_RUN_RECORD.template.json` | 换设备后另存的新环境与运行记录模板 |

模型仓库每个目录的 `BACKUP_PROVENANCE.json` 记录来源；π0/π0.5 还保留转换清单。原始计数、失败和成本证据随结果包保存。

## 2. 固定版本，不按未来的 main 漂移

| 内容 | 仓库 | 本次数据/代码提交 |
|---|---|---|
| 冻结代码基线 | `https://github.com/pumpkin601/FastRSI` | `d221e09ebd816521eddb9730965a61c3d5b9c338` |
| 最终权重 | `pumpkin601/FastRSI-final-weights` | `51bce91821b4eddc7e0f5a077c727a5c489e60bc` |
| 结果与依赖 | `pumpkin601/FastRSI-artifacts`，dataset | `1bb22c1db2ed3045c93024ebba417b3b45d404f0` |

本恢复记录另外保存在 GitHub 标签 **`backup-record-20261027-v1`**，也同步到两个 Hugging Face 仓库的 `reproduction/` 目录。上述数据提交刻意固定在这份说明加入之前，避免未来文档或数据变动造成版本混用。

三个仓库均为私有。新设备需登录有权限的 GitHub 和 Hugging Face 账号；当前所有者为 `pumpkin601`。访问凭据不包含在备份里。

## 3. 下载与校验

下载/校验工具使用 Python 3.12+、Git，以及 `huggingface_hub==2.3.0`。这只是恢复工具环境，不是模型运行环境。GitHub HTTPS 认证可先用 `gh auth login` 与 `gh auth setup-git` 完成。

选择一个新的工作目录后执行：

```bash
mkdir FastRSI-restore
cd FastRSI-restore

git clone --branch backup-record-20261027-v1 --single-branch https://github.com/pumpkin601/FastRSI.git record
git -C record worktree add --detach ../code d221e09ebd816521eddb9730965a61c3d5b9c338

python3 -m venv download-tools
./download-tools/bin/python -m pip install huggingface_hub==2.3.0
./download-tools/bin/hf auth login

./download-tools/bin/hf download pumpkin601/FastRSI-final-weights --revision 51bce91821b4eddc7e0f5a077c727a5c489e60bc --local-dir models
./download-tools/bin/hf download pumpkin601/FastRSI-artifacts --repo-type dataset --revision 1bb22c1db2ed3045c93024ebba417b3b45d404f0 --local-dir artifacts

./download-tools/bin/python record/reproduction/verify_restore.py --workspace .
```

模型约 79.71 GB，资料约 2.62 GB；首次完整哈希校验需要读取这些文件。全量解压后的归档独立内容合计约 13.89 GB，另有代码、文件系统开销及环境。因此建议至少预留 **150 GB 可用空间**；还要安装大型运行环境、保存新检查点时，建议 **200 GB 或更多**。这是容量规划，不是硬件性能保证。

如果先只下载资料，可以运行 `--scope artifacts` 校验；只校验模型用 `--scope models`。没有模型文件时，资料校验通过不代表模型已经恢复。

### 统一校验工具与历史清单

`artifacts/UPLOAD_MANIFEST.json` 是第一次备份的历史凭据，其中 README 的哈希对应早期说明。后续说明已更新，因此不要对当前组合快照直接运行旧 `verify_download.py` 来判断全部文件是否损坏。上面的统一工具按本记录固定的数据提交校验全部 30 个资料文件和 65 个模型文件；历史清单本身也会校验，原始证据不被重写。

## 4. 解压、保留去重关系

校验通过后执行：

```bash
./download-tools/bin/python record/reproduction/verify_restore.py --workspace . --extract
```

工具再次校验输入，然后分别解压到新的目录，避免不同历史版本相互覆盖：

```text
FastRSI-restore/
  record/                  本恢复记录及其版本历史
  code/                    固定代码提交，独立工作区
  models/                  六组最终模型及配置
  artifacts/               原始压缩包和凭据
  unpacked/
    reports/               报告、计数和成本证据
    original-libero/       已完成案例与代表性轨迹
    evaluation-source/     success_v2 冻结源码
    bounded-recovery/      选定恢复研究源码与最终凭据
    runtime/               冻结依赖、场景、视觉定位权重及补丁
```

五个解压目录必须尚不存在；中断产生的目录先另行保留，再选择新的输出位置。工具不会覆盖已有实验目录。

**保留硬链接非常重要。** 原版 LIBERO 的案例包逻辑大小约 544 GB，其中大量条目引用同一份成本日志；实际独立内容小得多。工具使用 tar 硬链接恢复去重关系。复制解压结果时也要保留硬链接，避免用会展开重复文件的工具。支持硬链接的本地文件系统更合适。各包的逻辑体积与独立内容体积都记录在 `BACKUP_RECORD.json`。

运行依赖的文件权限来自 `RUNTIME_FILE_MODES.json`；恢复工具会应用它，必要时拆开权限不同的硬链接。其他历史包按其原有归档权限恢复。

## 5. 把旧路径映射到新机器

旧项目根目录是 `/mnt/data/task/yuchen/FastPhyagentos`。不要仅做一次全局字符串替换：模型、源码、结果和运行依赖现在位于不同子目录。

使用 `PATH_MAP.json`，先匹配最长路径前缀，并将 `new_relative` 拼接到新工作目录：

- 六组模型分别位于 `models/<模型目录>`，保留原配置和归一化文件。
- 归档依赖位于 `unpacked/runtime/<原项目相对路径>`。
- success_v2 的主要源码和入口可从 `code/FastRSI/planning/libero_three_suites_20261024_success_v2/` 查找。
- 某些冻结来源另在 `unpacked/evaluation-source/` 或 `unpacked/bounded-recovery/`；用包内清单确认文件与哈希，不能只按类似文件名选版本。
- `metadata_only_reinstall_required` 的虚拟环境路径仅保留软件包元数据，不能当作已恢复的 Python 可执行环境。
- `/opt/hyhal`、`/etc/hfm`、`/opt/dtk/env.sh` 是旧主机依赖，需要在新平台重新确认。

把新路径、端口、设备编号和输出目录写入独立启动配置或适配补丁，记录改动与哈希；保留冻结源文件作为比较基线。旧 `owner.py`、资源绑定和调度文件含旧机器、GPU 与租约信息，不能把旧审计记录当作新机器可用状态直接执行。

## 6. 环境恢复边界

原运行环境包含 Linux amd64、HCU/DTK 和自定义镜像。macOS 可以下载、核验和查看结果，但无法直接运行这些 Linux/HCU 二进制。换成 NVIDIA/CUDA 时要重新安装匹配的框架与驱动，记录差异并重新验证。

| 已记录的信息 | 尚需在新设备完成 |
|---|---|
| 三个镜像的本地 image ID、标签、架构与大小 | 获取或重建兼容镜像；本地 image ID 不等于可从注册表拉取的 digest，归档未包含镜像层 |
| OpenVLA overlay：transformers 4.40.1、numpy 1.26.4 等 | 与对应基础环境组合验证 |
| X-VLA overlay：transformers 4.57.1 等 | 与对应基础环境组合验证，避免混装成一套环境 |
| Megapose 元数据位于 Python 3.8 环境；包括 numpy 1.23.5、Panda3D 1.10.14 | 重建兼容环境及外部框架依赖；元数据不是完整 lockfile |
| 冻结代码、模型、场景、补丁与版本校验值 | 路径映射、驱动/显存、模型加载、渲染和一个固定样例的实际验证 |

`ENVIRONMENT_PACKAGES.json` 按原目录分组；其中还包含包内部 vendor 元数据。不要把所有组直接合并为一个 pip requirements 文件。GPU 型号/显存的最低要求未在新机器实测，不能据当前记录承诺任意设备可运行。

## 7. 选择正确实验入口与续跑位置

| 工作 | 先读取的依据 |
|---|---|
| 原版 LIBERO 三套件 success_v2 | `code/FastRSI/planning/libero_three_suites_20261024_success_v2/RUN_PLAN.json`、`SOURCE_BINDINGS.json`、`ASSET_INDEX_FINAL.json` 及所选资源绑定文件 |
| FINAL48 有限恢复研究 | `FastRSI/planning/fastrsi_success_efficiency_20261018_v1/FINAL48_RECEIPT_20261018T025341Z.json`，位置见 bounded-recovery 包清单 |
| Object 剩余任务恢复研究 | `FastRSI/planning/libero_object_remaining7_20261021_v1/FINAL_COMPARISON_01.json`，以及 `runtime_bundle_v3/BUILD_INPUT_MANIFEST.json` |

success_v2 原始计划：12 个组，每组 10 个任务 × 50 个初始状态，seed=7；Spatial/Object/Goal 的步数上限分别为 220/280/300；最大 native replans=5，physical attempts=6。不要把这些配置套到别的研究，具体仍以对应冻结计划为准。

本次上传快照共有 **5,968 / 6,000** 个闭合案例。11 个组各 500 个，OpenVLA Goal 为 **468 / 500**，还差 32 个，身份列在 `PENDING_CASE_INDEX.jsonl`。`COMPLETED_CASE_INDEX.jsonl` 提供完成身份和终态记录位置；续跑前以 `(campaign, task_id, init_state_id)` 核对已经完成的案例，并核验相应 `TERMINAL_CASE.json`。这只说明已上传快照的进度，不代表此后服务器没有新结果。

**OpenVLA Goal/Spatial 专用权重未纳入这次选定模型备份。** 若要重新跑或补齐这些组，需先依据冻结配置找回并核对相应权重；不能用 LIBERO-10 或 Object 权重代替后仍称为同一实验。旧 Qwen2.5 和其他排除项同样按原研究依赖另行恢复。

在新机器先完成文件核验和环境检查，再选一个固定样例验证加载、渲染、动作与结果闭合。新复现运行使用独立 run ID、输出目录和新的运行记录；历史成功、失败、完整分母和成本保留，不能只重跑失败项后合并成更高成功率。原版 LIBERO 与 LIBERO-Pro、纯 VLA 与 P/R、不同预算和版本分别报告。

## 8. 每次继续时留记录

复制 `NEW_RUN_RECORD.template.json` 到新的运行目录，记录：机器与 GPU/驱动、Python/包版本、三个仓库提交、路径映射、适配补丁哈希、所用模型与场景哈希、实验参数、固定样例结果、已完成案例身份和新输出位置。

目前已验证的是云端备份内容与文件校验；**尚未在另一台设备实际恢复并重跑实验**。完成新机器的固定样例验证后，再把该运行结果和环境记录作为新的复现凭据保存。
