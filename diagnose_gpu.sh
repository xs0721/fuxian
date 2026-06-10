#!/bin/bash
echo "=========================================="
echo "GPU显存诊断报告"
echo "=========================================="

echo ""
echo "1. GPU基本信息："
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

echo ""
echo "2. 详细显存占用："
nvidia-smi

echo ""
echo "3. 所有使用GPU的进程："
nvidia-smi pmon -c 1

echo ""
echo "4. Python进程占用显存："
ps aux | grep python | grep -v grep

echo ""
echo "5. PyTorch显存分配："
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'可用GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'\nGPU {i}: {torch.cuda.get_device_name(i)}')
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(i)
        print(f'  总显存: {total/(1024**3):.2f} GB')
        print(f'  可用: {free/(1024**3):.2f} GB')
        print(f'  已用: {(total-free)/(1024**3):.2f} GB')
else:
    print('CUDA不可用')
"
