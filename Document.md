Windows 平台本地小体积 LLM 多语言翻译评测执行手册

本文档提供了一套完整的端到端自动化测试方案，用于在 Windows 本地环境下评估 8B 以下小模型的“所有主流语言互译”能力，并引入云端旗舰模型 DeepSeek V4 Flash 作为基准参照（Golden Baseline）。

1. 架构与工具链设计

本地推理引擎: llama.cpp 原生命令行工具 (无 HTTP 封装，0 额外开销，直接与 GPU 交互)。

云端基准 API: DeepSeek V4 Flash (提供高质量标准翻译结果)。

评测指标: SacreBLEU (chrF++) (针对多语种互译最公平的字符级 $n$-gram 指标)。

控制中枢: Python 3.10+ (负责数据调度、进程唤起、API 并发请求以及分数计算)。

2. Windows 环境准备

2.1 部署 llama.cpp 原生环境与模型

前往 llama.cpp 的 GitHub Releases 页面，下载最新的 Windows 预编译压缩包（通常名为 llama-bXXXX-bin-win-cuXXX-x64.zip，带 cu 的代表支持 Nvidia CUDA 加速）。

将压缩包解压到你指定的目录（例如 D:\llama_cpp），确保目录中存在 llama-cli.exe。

前往 Hugging Face 或 ModelScope 下载所需测试模型的 GGUF 文件（例如 qwen2.5-7b-instruct-q4_k_m.gguf）。将该模型文件放入 D:\llama_cpp 目录下。

2.2 配置 Python 测试环境

在 Windows 命令提示符中创建并激活虚拟环境，然后安装所需的依赖库（注意：不再需要安装 ollama 库）：

python -m venv llm_eval_env
.\llm_eval_env\Scripts\activate

# 安装评测库和 API 客户端
pip install openai sacrebleu pandas tqdm


2.3 准备 DeepSeek V4 API Key

在 DeepSeek 开放平台 获取你的 API Key。在 Windows PowerShell 中设置为环境变量：

$env:DEEPSEEK_API_KEY="sk-你的真实API密钥"


3. 测试数据集规范

创建一个 test_data.json 文件来存放测试用例（格式化参考 FLORES-200）：

[
  {
    "id": 1,
    "src_lang": "fr_XX",
    "tgt_lang": "zh_CN",
    "source": "Le chat dort sur le canapé.",
    "reference": "猫在沙发上睡觉。"
  },
  {
    "id": 2,
    "src_lang": "es_XX",
    "tgt_lang": "ja_JP",
    "source": "El sol brilla intensamente hoy.",
    "reference": "今日は太陽が明るく輝いています。"
  }
]


4. 核心 Python 自动化脚本

创建一个名为 run_benchmark.py 的文件。本脚本利用 subprocess 直接唤起 llama-cli.exe，llama.cpp 会将模型加载和运行日志输出到 stderr，而将干净的预测文本输出到 stdout，这天然契合 Python 的标准流捕获机制。

import os
import json
import subprocess
from openai import OpenAI
import sacrebleu
from tqdm import tqdm

# --- 核心配置区 ---
# 替换为你自己的实际路径
LLAMA_CLI_PATH = r"D:\llama_cpp\llama-cli.exe"
MODEL_PATH = r"D:\llama_cpp\qwen2.5-7b-instruct-q4_k_m.gguf"
DATASET_PATH = "test_data.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化 DeepSeek 客户端 (兼容 OpenAI SDK)
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="[https://api.deepseek.com](https://api.deepseek.com)"
)

# --- 核心函数 ---
def build_instruct_prompt(src_lang, tgt_lang, source_text):
    """
    针对指令跟随模型构建 ChatML 格式的 Prompt。
    使用原生的 llama-cli 时，直接输入完整的 token 格式可以省去解析麻烦。
    (以 Qwen 的 ChatML 格式为例)
    """
    system_msg = f"You are a precise professional translator. Translate the following text directly from {src_lang} to {tgt_lang}. Output the translation ONLY. Do not include any explanations, notes, or original text."
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{source_text}<|im_end|>\n<|im_start|>assistant\n"

def get_local_translation(prompt):
    """通过 subprocess 直接调起 llama.cpp CLI"""
    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "512",                 # 限制最大生成长度
        "-c", "2048",                # 上下文窗口
        "--temp", "0.1",             # 翻译任务保持低温度
        "-ngl", "99",                # 满载卸载到 GPU (根据你的 VRAM 调整)
        "--no-display-prompt"        # 不在 stdout 打印原始 prompt
    ]
    
    try:
        # capture_output 能够分离 stdout 和 stderr。llama.cpp 的加载日志全在 stderr，
        # 我们只需要干净的 stdout 文本。
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except Exception as e:
        print(f"执行 llama.cpp 失败: {e}")
        return ""

def get_deepseek_translation(src_lang, tgt_lang, source_text):
    """调用云端 DeepSeek V4 Flash 作为基准"""
    system_msg = f"You are a precise professional translator. Translate the following text directly from {src_lang} to {tgt_lang}. Output the translation ONLY."
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": source_text}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek API 出错: {e}")
        return ""

def clean_output(text):
    """剔除小模型可能会生成的引导废话"""
    noise_phrases = ["Here is the translation:", "The translation is:", "当然，这是翻译："]
    for phrase in noise_phrases:
        if text.startswith(phrase):
            text = text.replace(phrase, "", 1).strip()
    
    # 移除可能的闭合 token (以防模型打印出来了)
    text = text.replace("<|im_end|>", "").strip()
    return text

# --- 主执行流程 ---
def main():
    if not DEEPSEEK_API_KEY:
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量！")
        return
        
    if not os.path.exists(LLAMA_CLI_PATH) or not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到 llama-cli 或模型文件，请检查路径配置！")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc="Running Benchmark"):
        prompt = build_instruct_prompt(item['src_lang'], item['tgt_lang'], item['source'])
        
        # 跑本地原生进程
        local_pred = clean_output(get_local_translation(prompt))
        # 跑云端标准 API
        ds_pred = clean_output(get_deepseek_translation(item['src_lang'], item['tgt_lang'], item['source']))
        
        reference = item.get('reference') or ds_pred
            
        results.append({
            "id": item["id"],
            "src_lang": item["src_lang"],
            "tgt_lang": item["tgt_lang"],
            "source": item["source"],
            "reference": reference,
            "local_pred": local_pred,
            "deepseek_pred": ds_pred
        })

    print("\n=== 评测结果分析 ===")
    refs = [[r['reference'] for r in results]]
    sys_local = [r['local_pred'] for r in results]
    sys_ds = [r['deepseek_pred'] for r in results]

    chrf = sacrebleu.metrics.CHRF(word_order=2)
    local_Windows 平台本地小体积 LLM 多语言翻译评测执行手册

本文档提供了一套完整的端到端自动化测试方案，用于在 Windows 本地环境下评估 8B 以下小模型的多语言互译能力。

核心变更： 摒弃手动数据集，全面接入业界公认的多对多翻译黄金基准 FLORES-200，并以云端旗舰 DeepSeek V4 Flash 作为质量参照上限。

1. 架构与工具链设计

本地推理引擎: llama.cpp 原生命令行工具 (无 HTTP 封装，0 额外开销，直接与 GPU 交互)。

标准测试集: FLORES-200 (Meta 主导的 200 种语言平行语料库，通过 HF Datasets 动态拉取)。

云端基准 API: DeepSeek V4 Flash (提供高质量模型翻译上限基准)。

评测指标: SacreBLEU (chrF++) (针对多语种互译最公平的字符级 $n$-gram 指标)。

控制中枢: Python 3.10+ (负责数据调度、进程唤起、API 并发请求以及分数计算)。

2. Windows 环境准备

2.1 部署 llama.cpp 原生环境与模型

前往 llama.cpp 的 GitHub Releases 页面，下载最新的 Windows 预编译压缩包（通常名为 llama-bXXXX-bin-win-cuXXX-x64.zip，带 cu 的代表支持 Nvidia CUDA 加速）。

将压缩包解压到你指定的目录（例如 D:\llama_cpp），确保目录中存在 llama-cli.exe。

前往 Hugging Face 或 ModelScope 下载所需测试模型的 GGUF 文件（例如 qwen2.5-7b-instruct-q4_k_m.gguf）。将该模型文件放入 D:\llama_cpp 目录下。

2.2 配置 Python 测试环境

在 Windows 命令提示符中创建并激活虚拟环境，然后安装所需的依赖库（特别增加了 datasets 用于拉取官方测试集）：

python -m venv llm_eval_env
.\llm_eval_env\Scripts\activate

# 安装评测库、API 客户端及 HuggingFace 数据集工具
pip install openai sacrebleu pandas tqdm datasets


2.3 准备 DeepSeek V4 API Key

在 DeepSeek 开放平台 获取你的 API Key。在 Windows PowerShell 中设置为环境变量：

$env:DEEPSEEK_API_KEY="sk-你的真实API密钥"


3. 测试数据集规范 (FLORES-200)

我们不再使用手动编写的 JSON。工业界评测标准是使用 FLORES-200。
该数据集使用特定的 BCP-47 变体代码表示语言，常见的对应关系如下，你可以在 Python 脚本中自由组合它们进行互译测试：

英文: eng_Latn

简体中文: zho_Hans

日文: jpn_Jpan

法文: fra_Latn

德文: deu_Latn

西班牙文: spa_Latn

俄文: rus_Cyrl

4. 核心 Python 自动化脚本

创建一个名为 run_benchmark.py 的文件。该脚本会自动从 Hugging Face 下载 FLORES-200 指定语向的数据，切片后喂给本地 llama.cpp 和云端 DeepSeek 进行对抗评测。

import os
import json
import subprocess
from openai import OpenAI
import sacrebleu
from tqdm import tqdm
from datasets import load_dataset

# --- 核心配置区 ---
LLAMA_CLI_PATH = r"D:\llama_cpp\llama-cli.exe"
MODEL_PATH = r"D:\llama_cpp\qwen2.5-7b-instruct-q4_k_m.gguf"

# 定义你要测试的语向组合 (Source -> Target)
# FLORES-200 语言代码：英文(eng_Latn), 中文(zho_Hans), 日文(jpn_Jpan), 法文(fra_Latn), 德文(deu_Latn)
TEST_PAIRS = [
    ("eng_Latn", "zho_Hans"),  # 英译中
    ("fra_Latn", "jpn_Jpan"),  # 法译日 (考验小模型多语言中转能力)
    ("deu_Latn", "eng_Latn")   # 德译英
]

# 每个语向测试的句子数量。
# FLORES-200 devtest 完整有 1012 条，建议本地先测 50 条验证链路。
SAMPLES_PER_PAIR = 50 

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="[https://api.deepseek.com](https://api.deepseek.com)"
)

# --- 核心函数 ---
def build_instruct_prompt(src_lang, tgt_lang, source_text):
    """构建 ChatML 格式的指令"""
    # 故意将语言代码转换为自然语言名称，对通用模型更友好
    lang_map = {
        "eng_Latn": "English", "zho_Hans": "Simplified Chinese", 
        "jpn_Jpan": "Japanese", "fra_Latn": "French", "deu_Latn": "German"
    }
    src_str = lang_map.get(src_lang, src_lang)
    tgt_str = lang_map.get(tgt_lang, tgt_lang)
    
    system_msg = f"You are a precise professional translator. Translate the following text directly from {src_str} to {tgt_str}. Output the translation ONLY. Do not include any explanations."
    return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{source_text}<|im_end|>\n<|im_start|>assistant\n"

def get_local_translation(prompt):
    """通过 subprocess 调起 llama.cpp CLI"""
    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "512",
        "-c", "2048",
        "--temp", "0.1",
        "-ngl", "99",
        "--no-display-prompt"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except Exception as e:
        return f"[本地执行失败: {e}]"

def get_deepseek_translation(src_lang, tgt_lang, source_text):
    """调用云端 DeepSeek"""
    lang_map = {
        "eng_Latn": "English", "zho_Hans": "Simplified Chinese", 
        "jpn_Jpan": "Japanese", "fra_Latn": "French", "deu_Latn": "German"
    }
    system_msg = f"You are a precise professional translator. Translate the following text directly from {lang_map.get(src_lang, src_lang)} to {lang_map.get(tgt_lang, tgt_lang)}. Output the translation ONLY."
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": source_text}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[API 失败: {e}]"

def clean_output(text):
    noise_phrases = ["Here is the translation:", "The translation is:", "当然，这是翻译：", "以下是", "Translation:"]
    for phrase in noise_phrases:
        if text.startswith(phrase):
            text = text.replace(phrase, "", 1).strip()
    text = text.replace("<|im_end|>", "").strip()
    return text

def fetch_flores_data(src_lang, tgt_lang, limit):
    """利用 datasets 库自动拉取 FLORES-200 平行语料"""
    print(f"\n[*] 正在从 Hugging Face 获取 FLORES-200 数据集 ({src_lang} -> {tgt_lang})...")
    # trust_remote_code=True 是加载 FLORES 必需的
    src_dataset = load_dataset("facebook/flores", src_lang, split="devtest", trust_remote_code=True)
    tgt_dataset = load_dataset("facebook/flores", tgt_lang, split="devtest", trust_remote_code=True)
    
    data = []
    for i in range(min(limit, len(src_dataset))):
        data.append({
            "id": i,
            "source": src_dataset[i]["sentence"],
            "reference": tgt_dataset[i]["sentence"]
        })
    return data

# --- 主执行流程 ---
def main():
    if not DEEPSEEK_API_KEY:
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量！")
        return
        
    if not os.path.exists(LLAMA_CLI_PATH) or not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到 llama-cli 或模型文件！")
        return

    all_results = {}

    for src_lang, tgt_lang in TEST_PAIRS:
        print(f"\n==============================================")
        print(f" 开始评测语向: {src_lang} ===> {tgt_lang}")
        print(f"==============================================")
        
        pair_data = fetch_flores_data(src_lang, tgt_lang, SAMPLES_PER_PAIR)
        results = []

        for item in tqdm(pair_data, desc="推理进行中"):
            prompt = build_instruct_prompt(src_lang, tgt_lang, item['source'])
            
            local_pred = clean_output(get_local_translation(prompt))
            ds_pred = clean_output(get_deepseek_translation(src_lang, tgt_lang, item['source']))
            
            results.append({
                "id": item["id"],
                "source": item["source"],
                "reference": item["reference"],
                "local_pred": local_pred,
                "deepseek_pred": ds_pred
            })

        # 计算并打印当前语向的得分
        refs = [[r['reference'] for r in results]]
        sys_local = [r['local_pred'] for r in results]
        sys_ds = [r['deepseek_pred'] for r in results]

        chrf = sacrebleu.metrics.CHRF(word_order=2)
        local_score = chrf.corpus_score(sys_local, refs).score
        ds_score = chrf.corpus_score(sys_ds, refs).score
        
        print(f"\n>>> 语向 {src_lang}->{tgt_lang} 评测结果:")
        print(f"  [本地模型] chrF++: {local_score:.2f}")
        print(f"  [DeepSeek] chrF++: {ds_score:.2f}")
        print(f"  相对表现: {(local_score / max(ds_score, 1)) * 100:.1f}%\n")
        
        all_results[f"{src_lang}_to_{tgt_lang}"] = {
            "metrics": {
                "local_chrf": local_score,
                "deepseek_chrf": ds_score
            },
            "details": results
        }

    with open("flores_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n所有评测已完成，详细报告已保存至 flores_benchmark_report.json")

if __name__ == "__main__":
    main()
score = chrf.corpus_score(sys_local, refs)
    ds_score = chrf.corpus_score(sys_ds, refs) 

    print(f"【原生 llama.cpp 本地模型】平均 chrF++ 得分: {local_score.score:.2f}")
    print(f"【DeepSeek V4 Flash 云端基准】平均 chrF++ 得分: {ds_score.score:.2f}")
    print(f"-> 本地小模型达到了顶级旗舰模型能力的: {(local_score.score / max(ds_score.score, 1)) * 100:.1f}%\n")

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("详细结果已保存至 evaluation_report.json")

if __name__ == "__main__":
    main()
