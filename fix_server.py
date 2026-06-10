#!/usr/bin/env python3
"""
在服务器上直接运行此脚本来下载缺失的检查点文件
使用方法: python fix_server.py
"""
import os
import sys

def download_checkpoint():
    """下载 RewardModel 检查点文件"""

    # 检查点路径
    MULTIBIT_PROJECT = "/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master"
    CKPT_DIR = os.path.join(MULTIBIT_PROJECT, "ckpt")
    RM_HEADER_PATH = os.path.join(CKPT_DIR, "WatermarkDecoder-v_head.pt")

    print("=" * 60)
    print("修复 test9_multibit_watermark.py 缺失的检查点文件")
    print("=" * 60)
    print(f"\n目标路径: {RM_HEADER_PATH}\n")

    # 检查文件是否已存在
    if os.path.exists(RM_HEADER_PATH):
        print(f"✅ 检查点文件已存在，无需下载")
        print(f"   大小: {os.path.getsize(RM_HEADER_PATH) / 1024:.1f} KB")
        return True

    # 创建目录
    print(f"[1/3] 创建目录: {CKPT_DIR}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("     ✓ 目录已就绪\n")

    # 下载文件
    print("[2/3] 从 HuggingFace 下载检查点...")
    print("     仓库: xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b")
    print("     文件: v_head.pt\n")

    try:
        from huggingface_hub import hf_hub_download
        import shutil

        # 下载到临时位置
        print("     正在下载...")
        downloaded_path = hf_hub_download(
            repo_id="xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b",
            filename="v_head.pt",
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False
        )

        # 重命名为目标文件名
        target_path = RM_HEADER_PATH
        if downloaded_path != target_path and os.path.exists(downloaded_path):
            shutil.move(downloaded_path, target_path)

        print(f"     ✓ 下载完成\n")

    except ImportError:
        print("     ❌ 缺少 huggingface_hub 模块")
        print("     请先安装: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"     ❌ 下载失败: {e}")
        print("\n手动下载方法:")
        print("1. 访问: https://huggingface.co/xiaojunxu/WatermarkDecoder-Qwen2.5-1.5b/tree/main")
        print("2. 下载 v_head.pt 文件")
        print(f"3. 保存到: {RM_HEADER_PATH}")
        return False

    # 验证文件
    print("[3/3] 验证文件...")
    if os.path.exists(RM_HEADER_PATH):
        size_kb = os.path.getsize(RM_HEADER_PATH) / 1024
        print(f"     ✓ 文件已保存")
        print(f"     路径: {RM_HEADER_PATH}")
        print(f"     大小: {size_kb:.1f} KB")

        # 验证是否为有效的 PyTorch 文件
        try:
            import torch
            state_dict = torch.load(RM_HEADER_PATH, map_location='cpu', weights_only=True)
            print(f"     ✓ PyTorch 文件验证通过")
            print(f"     包含 {len(state_dict)} 个张量")
        except Exception as e:
            print(f"     ⚠️  文件验证警告: {e}")

        print("\n" + "=" * 60)
        print("✅ 修复完成！现在可以运行 test9_multibit_watermark.py")
        print("=" * 60)
        return True
    else:
        print("     ❌ 文件未找到")
        return False

if __name__ == "__main__":
    success = download_checkpoint()
    sys.exit(0 if success else 1)
