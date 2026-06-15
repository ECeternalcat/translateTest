# LLM 翻译评测框架

一键式本地 LLM 多语言翻译质量评测工具。基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 原生推理 + [FLORES-200](https://huggingface.co/facebook/flores) 标准数据集，支持 DeepSeek V4 Flash 云端基准对比。

## 快速开始

```
1. 将 llama-cli.exe 放入 bin/ 目录
2. 将 .gguf 模型文件放入 models/ 目录
3. 双击 setup.bat（仅首次）
4. 双击 run.bat 开始评测
```

评测结果输出至 `benchmark_report.json`。

## 目录结构

```
translateTest/
├── bin/                        # llama.cpp 可执行文件（llama-cli.exe）
│   └── .gitkeep
├── models/                     # GGUF 模型文件（*.gguf）
│   └── .gitkeep
├── config.json                 # 配置文件（语向、样本数、API Key 等）
├── test_data.json              # 本地测试用例（HF 不可用时自动降级）
├── setup.bat                   # 一键安装依赖
├── run.bat                     # 一键启动评测
├── run_benchmark.py            # 核心评测脚本
└── .gitignore
```

## 配置

编辑 `config.json`（所有字段均为可选）：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `llama_cli_path` | llama-cli.exe 路径 | 自动探测 `bin/` |
| `model_path` | GGUF 模型路径 | 自动探测 `models/` |
| `test_pairs` | 语向组合列表 | 英→中、法→日、德→英 |
| `samples_per_pair` | 每语向测试句数 | 50 |
| `deepseek_api_key` | DeepSeek API Key | 空（跳过云端对比） |
| `hf_token` | HuggingFace Token | 空（降级本地数据） |
| `skip_cloud_baseline` | 跳过云端对比 | false |
| `max_tokens` | 最大生成长度 | 512 |
| `temperature` | 采样温度 | 0.1 |
| `context_size` | 上下文窗口 | 2048 |
| `n_gpu_layers` | GPU 卸载层数 | 99 |

## 评测指标

使用 **SacreBLEU chrF++**（字符级 n-gram F-score），对多语种互译场景更公平。

## 环境要求

- Windows 10+
- Python 3.10+
- 支持 Vulkan 的显卡（NVIDIA / AMD / Intel 均可，CPU 推理亦可）
- DeepSeek API Key（可选，仅云端对比需要）
- HuggingFace Token（可选，仅 FLORES 在线拉取需要）

## 语言代码参考

| 语言 | FLORES 代码 |
|------|-------------|
| 英文 | `eng_Latn` |
| 简体中文 | `zho_Hans` |
| 日文 | `jpn_Jpan` |
| 法文 | `fra_Latn` |
| 德文 | `deu_Latn` |
| 西班牙文 | `spa_Latn` |
| 俄文 | `rus_Cyrl` |

## License

MIT
