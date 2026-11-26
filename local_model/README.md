# 本地 Qwen3.5 9B 模型服务

本路线在自己的 GPU 机器上运行 Qwen3.5 9B Q4_K_M，通过本地兼容接口供 Baseline / Explore 调用。最终课程实验在 Windows 的 RTX 4070 Ti SUPER 上完成。固定数据准备、六组实验命令和离线复算见[课程复现指南](../results/final_project/README.md)。下文保留早期多平台的部署说明。

## 1 环境与固定版本

在仓库根目录激活 `NLP_project`，按 [主 README](../README.md) 安装环境。Python 用于分词器、数据和实验；推理由原生 llama.cpp 完成，不要求安装 GPU 版 PyTorch。

| 项目 | 固定值 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3.5-9B` |
| 基础 revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| GGUF 来源 | `unsloth/Qwen3.5-9B-GGUF` |
| GGUF revision | `3885219b6810b007914f3a7950a8d1b469d598a5` |
| 文件 | `Qwen3.5-9B-Q4_K_M.gguf`，5,680,522,464 字节 |
| SHA256 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| 服务版本 | llama.cpp `b11871`，提交 `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` |
| 服务 alias | `Qwen3.5-9B-Q4_K_M` |

从 [llama.cpp b11871 官方发布页](https://github.com/ggml-org/llama.cpp/releases/tag/b11871) 下载适合操作系统及 GPU 后端的发行包，解压后让 `llama-server` 可执行文件在 PATH 中；Windows 对应 `llama-server.exe`。保留它依赖的动态库。原有实测来自 Windows + RTX 4070 Ti SUPER 16GB；当前 Mac 没有运行过本地 GGUF，不能沿用该硬件的测量结果。

## 2 下载模型与分词器

以下命令适用于 macOS/Linux 的 Bash；Windows PowerShell 可将多行命令合为一行执行。`hf` 随当前环境的 huggingface_hub 安装，[官方下载说明](https://huggingface.co/docs/huggingface_hub/guides/cli)支持按 revision 和文件名下载。

```bash
conda activate NLP_project
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf \
  --revision 3885219b6810b007914f3a7950a8d1b469d598a5 \
  --local-dir models/qwen35

# 仅下载相同基础 revision 的分词器
python api/setup_tokenizer.py
```

在正式使用前校验 GGUF：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
p = Path('models/qwen35/Qwen3.5-9B-Q4_K_M.gguf')
h = hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
assert p.stat().st_size == 5680522464
assert h.hexdigest() == '03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8'
print('GGUF checksum verified')
PY
```

## 3 创建本机配置

不要直接使用包含旧 `F:` 盘路径的 `Baseline/configs/searchqa.json`。以下代码保留原实验参数，生成两份独立本机配置：

```bash
python - <<'PY'
from pathlib import Path
import json
for dataset in ('searchqa', 'hotpotqa'):
    source = Path(f'Baseline/configs/{dataset}.json')
    target = Path(f'Baseline/configs/{dataset}_local.json')
    if target.exists():
        raise SystemExit(f'配置已存在，请检查后手工编辑：{target}')
    config = json.loads(source.read_text())
    config['backend'] = 'local'
    config['model'].update(
        tokenizer_path=str(Path('api/runtime/tokenizer').resolve()),
        precision='GGUF Q4_K_M; record actual GPU/backend in the experiment report',
        tokenizer_note='Pinned Qwen3.5-9B tokenizer; verify token counts against the serving GGUF',
    )
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(target)
PY
```

本地 `base_url` 是 `http://127.0.0.1:8000/v1`。默认回环服务不设密钥；如果自行开启服务鉴权，在进程环境设置 `PROJECT5_API_KEY`。本地模式不会自动加载根目录 `.env`。更换模型或服务版本时还需更新 `id/revision/precision/server_version` 和 tokenizer，不能只改地址。

## 4 启动 GPU 服务

在服务终端从仓库根目录运行。账本从分配开始计时，包括加载和空闲常驻，不是将请求延迟相加：

```bash
conda activate NLP_project
gpu_session="local-qwen35-$(date +%Y%m%d_%H%M%S)"
python Baseline/run.py gpu start \
  --ledger Baseline/runs/gpu_accounting.jsonl --session "${gpu_session}"

llama-server \
  --model models/qwen35/Qwen3.5-9B-Q4_K_M.gguf \
  --alias Qwen3.5-9B-Q4_K_M \
  --host 127.0.0.1 --port 8000 \
  --ctx-size 16384 --parallel 1 \
  --n-gpu-layers 99 --flash-attn on \
  --batch-size 512 --ubatch-size 256 \
  --jinja --reasoning off --no-context-shift
```

让服务终端保持运行，在另一个终端检查 `http://127.0.0.1:8000/health` 返回正常状态，`/v1/models` 包含配置的 alias。先确认本机 GPU 容量及实际加载情况；不能用一次服务启动成功代替真实 pilot。

服务退出或使用 Ctrl+C 停止后，在原服务终端关闭账本：

```bash
python Baseline/run.py gpu stop \
  --ledger Baseline/runs/gpu_accounting.jsonl --session "${gpu_session}"
```

当前公共 runner 对本地服务执行单 GPU、40 GPU 小时账本上限。`gpu stop` 只关闭账本，不停止服务进程，务必先实际停止服务。API 路线不使用这个账本。

## 5 跑本地 Baseline

服务保持运行，在另一个终端激活环境并从仓库根目录执行：

```bash
conda activate NLP_project
bash <<'BASH'
set -euo pipefail
run_tag="$(date +%Y%m%d_%H%M%S)-$$"
for dataset in searchqa hotpotqa; do
  config_path="Baseline/configs/${dataset}_local.json"
  data_dir="Baseline/data/prepared/${dataset}"
  run_dir="Baseline/runs/local/${dataset}/${run_tag}"
  python Baseline/download_data.py --dataset "${dataset}"
  python Baseline/run.py pilot --config "${config_path}" --data "${data_dir}" --output "${run_dir}/pilot"
  python Baseline/run.py initialize --config "${config_path}" --data "${data_dir}" --output "${run_dir}/shared_init"
  python Baseline/run.py static --condition N --config "${config_path}" --data "${data_dir}" --output "${run_dir}/N"
  python Baseline/run.py static --condition I --config "${config_path}" --data "${data_dir}" --output "${run_dir}/I"
  python Baseline/run.py train --config "${config_path}" --data "${data_dir}" --initialization "${run_dir}/shared_init" --output "${run_dir}/B"
done
BASH
```

查看本地进度：`python api/status.py --root Baseline/runs/local`。接口、结构、token 上限及数据边界与公共流程一致，输出进入 `runs/local`，不与 API 结果混用。

## 6 原 Windows 管理脚本

仓库原有 `start.ps1/status.ps1/stop.ps1/download_model.py/verify_service.py/run_pilot_measured.py` 是历史 Windows 部署脚本，仍保留原机绝对路径。若继续使用它们，先调整模型目录 `F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit`、Python `E:\Anaconda\envs\qwen3.5\python.exe` 和项目路径，并把固定版本服务二进制及 DLL 放入 `local_model/runtime/b11871/`。下载脚本额外依赖 `requests`。不要在当前 Mac 原样执行这些 Windows 脚本。

这些脚本的 `start.ps1` 会自动开启账本，`stop.ps1` 核对 PID 和模型后停止服务并关闭账本；使用它们时不要再手工重复执行 `gpu start`。旧 `requirements-lock.txt` 记录当时环境；当前 API 环境的锁文件在 `api/requirements-lock.txt`。
