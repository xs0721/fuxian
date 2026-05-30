"""测试运行器 — 独立运行任意水印攻击测试
用法:
  python run_test.py 1          # 运行测试1
  python run_test.py all        # 运行全部测试
  python run_test.py            # 列出可用测试
"""
import subprocess
import sys
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_SCRIPTS = {
    "1": "test1_physical_perturbation.py",
    "2": "test2_complex_rewrite.py",
    "3": "test3_watermark_stealing.py",
    "4": "test4_sira_targeted.py",
    "5": "test5_window_tradeoff.py",
    "6": "test6_kd_radioactivity.py",
    "7": "test7_b4_proxy_erasure.py",
    "8": "test8_adaptive_stealing.py",
    "9": "test9_multibit_watermark.py",
    "10": "test10_watermark_smoothing.py",
    "11": "test11_waterpark_fidelity.py",
    "12": "test12_demark_removal.py",
    "13": "test13_ditto_spoofing.py",
    "14": "test14_multikey_removal.py",
    "15": "test15_api_query_attack.py",
    "16": "test16_robustness_spoofing.py",
}

def run_test(num):
    script = TEST_SCRIPTS[num]
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return False
    print(f"\n{'='*60}\n运行: {script}\n{'='*60}")
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR,
                          env=os.environ.copy())
    return result.returncode == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("可用测试:")
        for k, v in TEST_SCRIPTS.items():
            exists = "✓" if os.path.exists(os.path.join(SCRIPT_DIR, v)) else "✗"
            print(f"  {exists} 测试{k}: {v}")
        print("\n用法: python run_test.py <编号>  或  python run_test.py all")
        sys.exit(0)

    arg = sys.argv[1]
    if arg == "all":
        failed = []
        for n in TEST_SCRIPTS:
            ok = run_test(n)
            if not ok:
                failed.append(n)
        if failed:
            print(f"\n以下测试失败: {failed}")
        else:
            print("\n全部测试完成!")
    else:
        run_test(arg)
