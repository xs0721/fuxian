"""智能模型选择器 - 根据系统资源自动选择最佳模型"""
import os
import sys
import torch
import platform
from typing import Dict, Tuple, Optional

# 强制 UTF-8 编码输出，解决 Windows GBK 乱码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 模型配置：(模型名, 最小显存GB, 推荐显存GB, 性能评分)
MODEL_CONFIGS = {
    # 小型模型（适合CPU或低显存）
    "facebook/opt-125m": {
        "min_vram": 0.5,
        "recommended_vram": 1.0,
        "performance_score": 1,
        "description": "OPT-125M (最小，适合快速测试)"
    },
    "facebook/opt-350m": {
        "min_vram": 1.0,
        "recommended_vram": 2.0,
        "performance_score": 2,
        "description": "OPT-350M (小型，基础性能)"
    },

    # 中型模型（推荐用于水印测试）
    "facebook/opt-1.3b": {
        "min_vram": 3.0,
        "recommended_vram": 6.0,
        "performance_score": 4,
        "description": "OPT-1.3B (中型，推荐用于水印测试)"
    },
    "facebook/opt-2.7b": {
        "min_vram": 6.0,
        "recommended_vram": 12.0,
        "performance_score": 5,
        "description": "OPT-2.7B (较大，性能较好)"
    },

    # 大型模型（高性能）
    "facebook/opt-6.7b": {
        "min_vram": 14.0,
        "recommended_vram": 20.0,
        "performance_score": 7,
        "description": "OPT-6.7B (大型，高性能)"
    },
    "facebook/opt-13b": {
        "min_vram": 28.0,
        "recommended_vram": 40.0,
        "performance_score": 8,
        "description": "OPT-13B (超大，最佳性能)"
    },

    # 其他高质量模型
    "meta-llama/Llama-2-7b-hf": {
        "min_vram": 14.0,
        "recommended_vram": 20.0,
        "performance_score": 8,
        "description": "Llama-2-7B (高质量，需要授权)"
    },
    "google/gemma-2-2b-it": {
        "min_vram": 5.0,
        "recommended_vram": 8.0,
        "performance_score": 6,
        "description": "Gemma-2-2B (高质量指令模型)"
    }
}


def get_gpu_memory() -> Optional[float]:
    """获取GPU显存大小（GB）"""
    if not torch.cuda.is_available():
        return None

    try:
        # 获取所有GPU的显存
        gpu_memories = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            gpu_memories.append(total_memory_gb)

        # 返回最大的GPU显存
        return max(gpu_memories) if gpu_memories else None
    except Exception as e:
        print(f"⚠️  获取GPU显存失败: {e}")
        return None


def get_available_memory() -> Optional[float]:
    """获取GPU当前可用显存（GB）"""
    if not torch.cuda.is_available():
        return None

    try:
        torch.cuda.empty_cache()
        # 获取主GPU的可用显存
        free_memory = torch.cuda.mem_get_info()[0]
        return free_memory / (1024**3)
    except Exception as e:
        print(f"⚠️  获取可用显存失败: {e}")
        return None


def check_model_cached(model_name: str, cache_dir: str = None) -> bool:
    """检查模型是否已完整缓存（包含权重文件）"""
    if cache_dir is None:
        if platform.system() == "Windows":
            cache_dir = "E:/Your_Cloud_Drive/hf_cache"
        else:
            cache_dir = "/root/autodl-tmp/hf_cache"

    if not os.path.exists(cache_dir):
        return False

    # 检查缓存目录中是否有该模型
    model_cache_name = model_name.replace("/", "--")
    potential_paths = [
        os.path.join(cache_dir, "models--" + model_cache_name),
        os.path.join(cache_dir, model_cache_name),
    ]

    for path in potential_paths:
        if os.path.exists(path):
            # 进一步检查是否有权重文件（确保模型完整）
            has_weights = False
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.pt')):
                        has_weights = True
                        break
                if has_weights:
                    break

            if has_weights:
                return True
    return False


def select_best_model(
    prefer_quality: bool = False,
    min_score: int = 1,
    max_score: int = 10,
    force_model: Optional[str] = None,
    prefer_cached: bool = True,
    cache_dir: str = None
) -> Tuple[str, Dict]:
    """
    根据系统资源智能选择最佳模型

    Args:
        prefer_quality: True=优先选择高性能模型（可能慢），False=优先选择能运行的模型
        min_score: 最低性能评分要求
        max_score: 最高性能评分限制
        force_model: 强制使用指定模型（忽略资源检查）
        prefer_cached: 优先选择已缓存的模型（避免下载）
        cache_dir: 模型缓存目录

    Returns:
        (model_name, model_config)
    """

    # 如果强制指定模型
    if force_model:
        if force_model in MODEL_CONFIGS:
            config = MODEL_CONFIGS[force_model]
            print(f"🔧 使用强制指定的模型: {force_model}")
            print(f"   {config['description']}")
            return force_model, config
        else:
            print(f"⚠️  未知模型: {force_model}，回退到自动选择")

    # 检查环境变量配置
    env_model = os.environ.get("WATERMARK_MODEL")
    if env_model and env_model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[env_model]
        print(f"🔧 使用环境变量指定的模型: {env_model}")
        print(f"   {config['description']}")
        return env_model, config

    # 获取GPU信息
    total_vram = get_gpu_memory()
    available_vram = get_available_memory()

    if total_vram is None:
        # CPU模式，使用最小模型
        print("💻 未检测到GPU，使用CPU模式")
        model_name = "facebook/opt-125m"
        return model_name, MODEL_CONFIGS[model_name]

    print(f"🎮 GPU显存: 总计 {total_vram:.1f}GB, 当前可用 {available_vram:.1f}GB")

    # 筛选符合条件的模型
    suitable_models = []
    cached_models = []

    for model_name, config in MODEL_CONFIGS.items():
        # 检查性能评分范围
        if config["performance_score"] < min_score or config["performance_score"] > max_score:
            continue

        # 检查显存要求（严格 + 安全裕度）
        # 添加60%安全裕度，预留推理开销、峰值和碎片
        # 推理峰值约为模型大小的1.3-1.5倍，60%裕度更保守
        safety_margin = 1.6

        if prefer_quality:
            # 质量优先：使用推荐显存 + 安全裕度
            required_vram = config["recommended_vram"] * safety_margin
            if available_vram < required_vram:
                continue
        else:
            # 兼容优先：使用最小显存 + 安全裕度
            required_vram = config["min_vram"] * safety_margin
            if available_vram < required_vram:
                continue

        # 显存充足，加入候选
        suitable_models.append((model_name, config))

        # 检查是否已缓存
        if check_model_cached(model_name, cache_dir):
            cached_models.append((model_name, config))

    if not suitable_models:
        # 没有合适的模型，回退到最小模型
        print("⚠️  显存不足，使用最小模型")
        model_name = "facebook/opt-125m"
        return model_name, MODEL_CONFIGS[model_name]

    # 优先使用已缓存的模型（避免下载）
    if prefer_cached and cached_models:
        print("📦 优先使用已缓存且显存充足的模型")
        cached_models.sort(key=lambda x: x[1]["performance_score"], reverse=True)
        best_model, best_config = cached_models[0]
        cache_status = "✅ 已缓存"
    else:
        # 按性能评分排序，选择最好的
        suitable_models.sort(key=lambda x: x[1]["performance_score"], reverse=True)
        best_model, best_config = suitable_models[0]
        cache_status = "📦 已缓存" if check_model_cached(best_model, cache_dir) else "⬇️  需要下载"

    print(f"✅ 自动选择模型: {best_model}")
    print(f"   {best_config['description']}")
    print(f"   性能评分: {best_config['performance_score']}/10")
    print(f"   显存需求: 最小{best_config['min_vram']:.1f}GB / 推荐{best_config['recommended_vram']:.1f}GB")
    print(f"   缓存状态: {cache_status}")

    return best_model, best_config


def get_model_for_test(test_name: str = "", cache_dir: str = None) -> str:
    """
    return "facebook/opt-1.3b"  # 强制 OPT，避免 Gemma2 CUDA 问题
    为测试获取合适的模型（便捷接口）

    使用优先级：
    1. 环境变量 WATERMARK_MODEL
    2. 环境变量 PREFER_QUALITY=1 时选择高质量模型
    3. 自动根据显存选择（优先使用已缓存的模型）

    示例:
        export WATERMARK_MODEL=facebook/opt-1.3b  # 强制使用特定模型
        export PREFER_QUALITY=1  # 优先使用高质量模型（如果显存够）
        export MAX_MODEL_SCORE=7  # 允许更大的模型（默认5）
    """
    prefer_quality = os.environ.get("PREFER_QUALITY", "0") == "1"
    # 允许通过环境变量控制最大模型大小（默认10=不限制）
    max_model_score = int(os.environ.get("MAX_MODEL_SCORE", "10"))

    if test_name:
        print(f"\n{'='*60}")
        print(f"为 {test_name} 选择模型...")
        print(f"{'='*60}")

    model_name, config = select_best_model(
        prefer_quality=prefer_quality,
        min_score=1,
        max_score=max_model_score,  # 默认10（不限制），可通过环境变量调整
        prefer_cached=True,  # 优先使用已缓存的模型
        cache_dir=cache_dir
    )

    # 显示使用提示
    if config["performance_score"] <= 2:
        print(f"\n💡 提示: 当前使用小型模型，测试效果可能有限")
        print(f"   如需更好效果，可以:")
        print(f"   1. 设置环境变量: export PREFER_QUALITY=1")
        print(f"   2. 指定模型: export WATERMARK_MODEL=facebook/opt-1.3b")
        print(f"   3. 升级GPU获得更大显存")

    return model_name


def list_available_models(show_all: bool = False):
    """列出所有可用模型及其要求"""
    total_vram = get_gpu_memory()
    available_vram = get_available_memory()

    print("\n" + "="*80)
    print("可用模型列表")
    print("="*80)

    if total_vram:
        print(f"当前系统: GPU显存 {total_vram:.1f}GB (可用 {available_vram:.1f}GB)")
    else:
        print("当前系统: CPU模式")

    print("\n" + "-"*80)
    print(f"{'模型名称':<40} {'性能':<6} {'显存需求':<20} {'状态':<10}")
    print("-"*80)

    for model_name, config in sorted(MODEL_CONFIGS.items(),
                                     key=lambda x: x[1]["performance_score"]):
        score = config["performance_score"]
        vram_req = f"{config['min_vram']:.1f}~{config['recommended_vram']:.1f}GB"

        # 判断是否可用
        if total_vram is None:
            status = "⚠️  需要GPU" if config["min_vram"] > 1 else "✅ 可用"
        elif available_vram >= config["recommended_vram"]:
            status = "✅ 推荐"
        elif available_vram >= config["min_vram"]:
            status = "⚠️  可用"
        else:
            status = "❌ 显存不足"

        # 只显示可用的或全部
        if show_all or "✅" in status or "⚠️" in status:
            print(f"{model_name:<40} {score:>2}/10  {vram_req:<20} {status:<10}")

    print("-"*80)
    print(f"\n{'='*80}")
    print("使用方法:")
    print("  export WATERMARK_MODEL=facebook/opt-1.3b  # 指定模型")
    print("  export PREFER_QUALITY=1                  # 优先使用高质量模型")
    print("="*80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_available_models(show_all=True)
    else:
        # 测试模型选择（与实际使用保持一致）
        print("\n=== 自动选择模式 ===")
        model1, _ = select_best_model(prefer_quality=False)
        print(f"\n选择的模型（兼容优先）: {model1}")

        print("\n=== 质量优先模式 ===")
        model2, _ = select_best_model(prefer_quality=True)
        print(f"\n选择的模型（质量优先）: {model2}")

        print("\n运行 'python model_selector.py list' 查看所有可用模型")
