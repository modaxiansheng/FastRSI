# FastRSI 最终有效版本备份

本次在 2026-10-26 的源码、结果备份上补充最终方案依赖的模型和运行资源。原始服务器文件未修改，未启动评测或训练。

## 三个私有仓库

- 代码和入口：https://github.com/pumpkin601/FastRSI
- 报告、逐例证据和运行依赖：https://huggingface.co/datasets/pumpkin601/FastRSI-artifacts
- 模型权重：https://huggingface.co/pumpkin601/FastRSI-final-weights

## 保留的六组主模型

| 目录 | 用途 | 大约大小 |
|---|---|---|
| pi05_libero_pytorch_fp32 | 已使用的 π0.5 LIBERO 转换版本，保持 FP32 | 14.47 GB |
| pi0_libero_pytorch_fp32 | 已使用的 π0 LIBERO 转换版本，保持 FP32 | 14.01 GB |
| xvla_libero | 已使用的 X-VLA LIBERO | 3.52 GB |
| openvla_libero10 | 已使用的 OpenVLA LIBERO-10 | 15.09 GB |
| openvla_object | 已归档成功 Object 恢复方案的专用策略 | 15.09 GB |
| qwen3vl8b-shared-cache-v1_20261007 | 已使用的 Qwen3-VL-8B-Instruct | 17.55 GB |

“最终”指本次筛选的实际实验依赖，不代表这些公共基础模型由 FastRSI 训练。每个目录有原始配置、必要 tokenizer/归一化文件和来源记录。公共仓库复制前对服务器原文件重新计算 SHA256，并与固定源版本的文件哈希比较；目的仓库再按文件大小和 SHA256/Git blob 哈希核验。

旧 Qwen2.5、OpenVLA Goal/Spatial、无用中间检查点、失败开发候选和重复媒体不纳入此增补。依赖这些排除项的历史基线不保证仅凭此备份离线复现。正式评测失败计数、完整分母和成本证据仍保留。

## 恢复步骤

1. 克隆 GitHub 私有仓库；通过 `hf auth login` 登录有权限的 Hugging Face 账号。
2. 在有至少约 90 GB 可用空间的新目录下载模型：
   `hf download pumpkin601/FastRSI-final-weights --local-dir final-weights`
3. 下载数据仓库：
   `hf download pumpkin601/FastRSI-artifacts --repo-type dataset --local-dir artifacts`
4. 使用仓库内 `verify_weights.py`（下载校验工具）校验模型。对数据包按各自上传清单及 SHA256 校验，再解压到新目录。原始结果包使用原有 `verify_download.py`；增补运行包运行 `python verify_runtime.py` 校验，或使用 Python 3.12+ 运行 `python verify_runtime.py --extract restored-runtime`。
5. `final-runtime-dependencies-20261026.tar.gz` 按服务器项目相对路径保存冻结依赖、场景、Megapose 权重和补丁。包清单给出原始及上传字节哈希，`RUNTIME_FILE_MODES.json` 记录需恢复的文件权限，恢复脚本会一并处理。原始绝对路径需在新机器映射到新根目录，不能直接照搬旧路径运行。
6. Python 环境保留 distribution 元数据，而不是完整虚拟环境。需要按平台重新安装依赖；主机 HCU/DTK 驱动、容器镜像和系统运行库需另外准备。冻结覆盖层内的二进制扩展也需要兼容平台。
7. 以已归档成功版本的 `SOURCE_BINDINGS.json`、资源绑定文件和各运行包说明选择入口。下载校验通过只说明备份字节完整，不等于已在新机器完成运行验证。本次没有重新运行 GPU 评测。

保存标准是“最终可用方案及其必要依赖”，不是服务器全盘镜像。
