"""
LLM 多语言翻译评测脚本 (便携版)
=================================
将 llama-cli.exe 放入 bin/ 目录，将 .gguf 模型放入 models/ 目录，
双击 run.bat 即可自动完成评测。

可选: 在 config.json 的 deepseek_api_key 中填写 API Key，
      或设置 DEEPSEEK_API_KEY 环境变量以启用云端基准对比。
      或编辑 config.json 自定义语向、样本数等参数。
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

# 确保中文输出在各种终端下正常显示
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
import sacrebleu
from tqdm import tqdm
from datasets import load_dataset


# ============================================================
# 路径工具 — 自动探测
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


def load_config():
    """加载 config.json (所有字段可选)"""
    cfg_path = BASE_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_llama_cli():
    """
    按优先级查找 llama-cli.exe:
    1. config.json 中指定的路径
    2. bin/llama-cli.exe
    3. ./llama-cli.exe (脚本同目录)
    """
    cfg = load_config()
    user_path = cfg.get("llama_cli_path", "")
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p.resolve())

    candidates = [
        BASE_DIR / "bin" / "llama-cli.exe",
        BASE_DIR / "llama-cli.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def find_model():
    """
    按优先级查找 .gguf 模型:
    1. config.json 中指定的路径
    2. models/ 目录下第一个 .gguf 文件
    3. 脚本同目录下第一个 .gguf 文件
    """
    cfg = load_config()
    user_path = cfg.get("model_path", "")
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p.resolve())

    candidates = [
        BASE_DIR / "models",
        BASE_DIR,
    ]
    for d in candidates:
        if d.is_dir():
            gguf_files = sorted(d.glob("*.gguf"))
            if gguf_files:
                return str(gguf_files[0].resolve())
    return None


# ============================================================
# 配置加载
# ============================================================
CFG = load_config()

LLAMA_CLI_PATH = find_llama_cli()
MODEL_PATH = find_model()

TEST_PAIRS = CFG.get("test_pairs", [
    ("eng_Latn", "zho_Hans"),
    ("fra_Latn", "jpn_Jpan"),
    ("deu_Latn", "eng_Latn"),
])
SAMPLES_PER_PAIR = CFG.get("samples_per_pair", 50)
SKIP_CLOUD = CFG.get("skip_cloud_baseline", False)

MAX_TOKENS = CFG.get("max_tokens", 512)
TEMPERATURE = CFG.get("temperature", 0.1)
CONTEXT_SIZE = CFG.get("context_size", 2048)
N_GPU_LAYERS = CFG.get("n_gpu_layers", 99)

# HF Token: 优先环境变量，其次 config.json
HF_TOKEN = os.getenv("HF_TOKEN", "") or CFG.get("hf_token", "")

# API Key: 优先环境变量，其次 config.json
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "") or CFG.get("deepseek_api_key", "")

LANG_MAP = {
    "eng_Latn": "English",
    "zho_Hans": "Simplified Chinese",
    "jpn_Jpan": "Japanese",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "rus_Cyrl": "Russian",
}


# ============================================================
# 启动前校验
# ============================================================
def ensure_dirs():
    """自动创建缺失的目录"""
    for d in [BASE_DIR / "bin", BASE_DIR / "models"]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 已自动创建目录: {d.name}/")


def preflight():
    """启动前检查，给出清晰的中文提示"""
    ensure_dirs()

    ok = True

    if LLAMA_CLI_PATH is None:
        print("=" * 50)
        print("  ⚠ 未找到 llama-cli.exe")
        print(f"  请将 llama-cli.exe 放入: {BASE_DIR / 'bin'}")
        print("  或在 config.json 中设置 \"llama_cli_path\"")
        print("=" * 50)
        ok = False
    else:
        print(f"  ✓ llama-cli: {LLAMA_CLI_PATH}")

    if MODEL_PATH is None:
        print("=" * 50)
        print("  ⚠ 未找到 .gguf 模型文件")
        print(f"  请将 .gguf 模型文件放入: {BASE_DIR / 'models'}")
        print("  或在 config.json 中设置 \"model_path\"")
        print("=" * 50)
        ok = False
    else:
        print(f"  ✓ 模型文件:  {MODEL_PATH}")

    return ok


# ============================================================
# 核心函数
# ============================================================
def build_instruct_prompt(src_lang, tgt_lang, source_text):
    src_str = LANG_MAP.get(src_lang, src_lang)
    tgt_str = LANG_MAP.get(tgt_lang, tgt_lang)

    system_msg = (
        f"You are a precise professional translator. "
        f"Translate the following text directly from {src_str} to {tgt_str}. "
        f"Output the translation ONLY. Do not include any explanations."
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{source_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_local_translation(prompt):
    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(MAX_TOKENS),
        "-c", str(CONTEXT_SIZE),
        "--temp", str(TEMPERATURE),
        "-ngl", str(N_GPU_LAYERS),
        "--no-display-prompt",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[本地执行失败: {e}]"


def get_deepseek_translation(src_lang, tgt_lang, source_text):
    if not DEEPSEEK_API_KEY:
        return "[未配置 API Key]"

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    system_msg = (
        f"You are a precise professional translator. "
        f"Translate the following text directly from "
        f"{LANG_MAP.get(src_lang, src_lang)} to "
        f"{LANG_MAP.get(tgt_lang, tgt_lang)}. "
        f"Output the translation ONLY."
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": source_text},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[API 失败: {e}]"


def clean_output(text):
    noise_phrases = [
        "Here is the translation:",
        "The translation is:",
        "当然，这是翻译：",
        "以下是",
        "Translation:",
    ]
    for phrase in noise_phrases:
        if text.startswith(phrase):
            text = text.replace(phrase, "", 1).strip()
    return text.replace("<|im_end|>", "").strip()


def fetch_flores_data(src_lang, tgt_lang, limit):
    """尝试从 HF 拉取 FLORES-200，失败则降级到本地 test_data.json"""
    load_kwargs = {"split": "devtest"}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    try:
        print(f"\n[*] 正在从 Hugging Face 获取 FLORES-200 数据集 "
              f"({src_lang} -> {tgt_lang})...")
        src_dataset = load_dataset(
            "facebook/flores", src_lang, **load_kwargs
        )
        tgt_dataset = load_dataset(
            "facebook/flores", tgt_lang, **load_kwargs
        )

        data = []
        for i in range(min(limit, len(src_dataset))):
            data.append({
                "id": i,
                "source": src_dataset[i]["sentence"],
                "reference": tgt_dataset[i]["sentence"],
            })
        return data

    except Exception as e:
        print(f"  ⚠ FLORES 获取失败: {e}")
        print(f"  → 降级使用本地 test_data.json")
        return _load_local_test_data(limit)


def _load_local_test_data(limit):
    """从本地 test_data.json 加载测试用例"""
    local_path = BASE_DIR / "test_data.json"
    if not local_path.exists():
        print("  ⚠ 本地 test_data.json 也不存在！")
        return []

    with open(local_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for i, item in enumerate(raw[:limit]):
        data.append({
            "id": item.get("id", i),
            "source": item["source"],
            "reference": item["reference"],
        })
    print(f"  ✓ 从本地加载了 {len(data)} 条测试用例")
    return data


# ============================================================
# 主执行流程
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  LLM 多语言翻译评测框架")
    print("=" * 60 + "\n")

    # 启动前校验
    if not preflight():
        print("\n请解决以上问题后重新运行。")
        input("\n按 Enter 键退出...")
        sys.exit(1)

    # 云端基准状态
    if SKIP_CLOUD:
        print("  - 云端基准: 已跳过 (config.json skip_cloud_baseline=true)")
    elif not DEEPSEEK_API_KEY:
        print("  - 云端基准: 跳过 (未设置 API Key，可在 config.json 的 deepseek_api_key 中填写)")
    else:
        print("  - 云端基准: DeepSeek V4 Flash ✓")

    print(f"\n  语向数量: {len(TEST_PAIRS)}")
    print(f"  每语向样本数: {SAMPLES_PER_PAIR}")
    print(f"  模型: {Path(MODEL_PATH).name}\n")

    all_results = {}

    for src_lang, tgt_lang in TEST_PAIRS:
        print(f"\n{'─' * 50}")
        print(f"  语向: {src_lang} → {tgt_lang}")
        print(f"{'─' * 50}")

        pair_data = fetch_flores_data(src_lang, tgt_lang, SAMPLES_PER_PAIR)
        results = []

        for item in tqdm(pair_data, desc="  推理中"):
            prompt = build_instruct_prompt(
                src_lang, tgt_lang, item["source"]
            )

            local_pred = clean_output(get_local_translation(prompt))
            ds_pred = "" if SKIP_CLOUD else clean_output(
                get_deepseek_translation(
                    src_lang, tgt_lang, item["source"]
                )
            )

            results.append({
                "id": item["id"],
                "source": item["source"],
                "reference": item["reference"],
                "local_pred": local_pred,
                "deepseek_pred": ds_pred if ds_pred else "(skipped)",
            })

        # 计算 chrF++ 得分
        refs = [[r["reference"] for r in results]]
        sys_local = [r["local_pred"] for r in results]

        chrf = sacrebleu.metrics.CHRF(word_order=2)
        local_score = chrf.corpus_score(sys_local, refs).score

        print(f"\n  ┌─────────────────────────────")
        print(f"  │ 语向: {src_lang} → {tgt_lang}")
        print(f"  │ 本地模型 chrF++: {local_score:.2f}")

        if not SKIP_CLOUD and DEEPSEEK_API_KEY:
            sys_ds = [r["deepseek_pred"] for r in results]
            ds_score = chrf.corpus_score(sys_ds, refs).score
            ratio = (local_score / max(ds_score, 1)) * 100
            print(f"  │ DeepSeek  chrF++: {ds_score:.2f}")
            print(f"  │ 相对表现: {ratio:.1f}%")
            all_results[f"{src_lang}_to_{tgt_lang}"] = {
                "metrics": {
                    "local_chrf": local_score,
                    "deepseek_chrf": ds_score,
                },
                "details": results,
            }
        else:
            all_results[f"{src_lang}_to_{tgt_lang}"] = {
                "metrics": {"local_chrf": local_score},
                "details": results,
            }

        print(f"  └─────────────────────────────")

    report_path = BASE_DIR / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"  评测完成！报告已保存至:")
    print(f"  {report_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
