#!/bin/bash

# 修改 test10-17 的 X 轴旋转
for file in test10_watermark_smoothing.py test11_waterpark_fidelity.py test12_demark_removal.py \
            test13_ditto_spoofing.py test14_multikey_removal.py test15_api_query_attack.py \
            test16_robustness_spoofing.py test17_watermark_learnability.py; do
    
    if [ -f "$file" ]; then
        echo "处理: $file"
        
        # 在 sns.boxplot 后添加 X 轴旋转代码
        # 查找是否已经有旋转代码
        if ! grep -q "tick_params.*rotation.*45" "$file"; then
            # 备份
            cp "$file" "${file}.bak"
            
            # 添加旋转（在 legend 行之后）
            sed -i '/ax.*\.legend(/a\    # X 轴标签旋转\n    ax.tick_params(axis=\"x\", rotation=45)\n    for label in ax.get_xticklabels():\n        label.set_rotation(45)\n        label.set_ha(\"right\")' "$file"
            
            echo "  ✅ 已添加 X 轴旋转"
        else
            echo "  ⏭️  已有旋转代码"
        fi
    else
        echo "  ⚠️  文件不存在: $file"
    fi
done

echo "完成！"
