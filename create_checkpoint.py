#!/usr/bin/env python3
"""
在服务器上直接运行此脚本来创建检查点文件
使用方法: python create_checkpoint.py
"""
import os
import base64

# Base64 编码的检查点文件
CHECKPOINT_B64 = """UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAVgGAAAAd2VpZ2h0cQJjdG9yY2guX3V0aWxzCl9yZWJ1aWxkX3RlbnNvcl92MgpxAygoWAcAAABzdG9yYWdlcQRjdG9yY2gKQkZsb2F0MTZTdG9yYWdlCnEFWAEAAAAwcQZYAwAAAGNwdXEHTQAGdHEIUUsASwFNAAaGcQlNAAZLAYZxColoAClScQt0cQxScQ1zfXEOWAkAAABfbWV0YWRhdGFxD2gAKVJxEFgAAAAAcRF9cRJYBwAAAHZlcnNpb25xE0sBc3NzYi5QSwcIhscqGusAAADrAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARABYAYXJjaGl2ZS9ieXRlb3JkZXJGQhIAWlpaWlpaWlpaWlpaWlpaWlpabGl0dGxlUEsHCIU94xkGAAAABgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA+AGFyY2hpdmUvZGF0YS8wRkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrKvDY8jbyuPKa8ITytvKy8JbwWvC880Dyuu/S7bjyDvMq8tDx5PGa8rbwMO7i7HztjvKc8MDxfPHY8PLyHvAU8ajthPL+8Vrvcuzc8dTy0PAq8mzxBu5o89rvBvBE7KDyjvB27szsVvMq8vzzbO8S7gzuHO1a8mru5u7W7nbrwuhI8cTuovAa8cbyavPW6sLyLPIY8pjzGvKM8UDxPPIQ8cDtJPM48e7ukvMw8BzzcOh08KDy2vFe8GjzKPLQ7XDv/uxE8hjsZPKI8zzxOvHS8zbzLvL68GrwKvJ67k7yEPOc78bu9O4c8ZzyDu7y8BLwxPJ28Obt/PH08rDsIvMe8mDydutK7arx5vEs8Rbxhu8m79butvIC8SDxwOxK8GTz9ugg73jpwvNy7gTxmPKI7dzy9u9A79btSPJW8/bsUPEk8uLwePMu8FryDO628wzvau2u8zrsIPLI8u7uUvLa8v7xLO828TrzQvDu8nrvAPAq8PjwDvB88SbyhvH28GbuCvK67vjzAPHi6orzEO6M7rTy/O148QbyfvJS8LbztO708CrtAOfC6hDyBO5q86TtavCo8hLyuPKE8irwOO5i8szr7uuq7CjzGvAO8u7xsPFm7mjyBu3a8n7xwvEK8vbzKvHO8vDuGu5g8QzwUO/W5ujvPO9E7H7zwu6M7uryQutu7eDzGO268tDvXOyw7bbw2PPK7mby+O8u8xjtOO7U8QjwjvHA81bs7u9A8hDzyu6a8iLwEvI07RbzJvAQ8gzxMPMQ8ADwpO0Q8nLz9u208tbw2vHQ8DzyxPAO7a7ydu4u8x7yRu2c8rrwrPEk8D7wBvAY8Cjw5vMk8IrznO5W8Njydu8S8iLy1PLY8qLyYPIK7WLuTPJ28ELzGPDc8vbzQvCE8Q7whO8g8pDwIvMG8GruxvLy8hbuLPHe7GjwCOx88v7saPMI8R7zKPAe8rjnPPEa8fDzDvF07NLy3vPg6g7zLvO67pTyCvB+8KDy+vL+8v7qXvMQ8oDwhu0Y8vLwRPGw8UrvBPLc8s7z1u0a8GLwdvIc8YDuUPMg8B7zCvFY8HDsYvMM7FbwKPLQ8zTq+PDo84LszOd67w7xzu5s8Zrvqu5S76rtvPJi8xbyqPM67JzyTvMs8XTu7PLy8hjwZvJo7abywPLK7ojyGvKC8qbseu4K6y7yIurw8Q7yGPF88M7xJvDC7FjuFO2u8GTuvuoQ8zDt3uxQ8Izz/O3A6yzx9PJw8HzwEvHG8hjysOQ48TjyWuws8y7pNvKu8B7z3O5O7yLwRvJY81jsvPCY7qzyIPEy8zTxPvM87pjs4PIo83buXvIG8yLy/vGe8k7yXvIO80LyjO8m7frySukC6G7xHPHa8uzyBu7o8yzyAvA07A7yhPJe8KLsdvNC8sTyTvEC8FDx6vIW8Gjt6Oou6XzyEvMa7q7y6PBo8kLy5vC68rjwPu+U7hzqWvHC8wTyEPGc8oLsVOrW8ADybvKc8y7vFPLA8BzqeOsA8PbwMPIg8yDwSvBi6E7ygPDm8fryKPCk8rDzMvEY8LTxVPB07HLsUPHg80Lxxuwq8MLzYO5e8r7y5PIS8WTy+uze8ojypvPk7QDx0u8A8xLyKPNA8erwDPKC8sjwRPHs6O7z4O4Y8GTypPP47rLwBvM48xryVvLO8pTyPPLu8SzxHPMU8NrvDO1m8hjzXO387ZjzXO4+8NbxAOzO8sLyIPOM79DoEvHU8ITydPCG87TsYPB68kbzju/+7vbwTvLW8IrwfPM670Dx0PLU8nTwNO7W8kTz4Oko8HbwjPDK86ruOvO27tLxMvLg8JDy5u8A8uzxDuvy7QTyjO0s7+DuqPAY8dDyOvMS76TtbPBA7mjyruss7hrxUvPk72bu+vKM8hzyNPFm8oDyovNc6xryOPBC8x7zEvNA8NryrvB85iznwuly8XjygPCA8p7w6vIu8ZztkPK65SjyNPBG8TrwrvIA8V7w6vGq8ObyOvCM8o7xsvJc8fjy9PNA8r7yFPMu8Vzyvu508krugu0y8jrsFPAG8q7w1PD+8qDz6uJc8ozwNvII88TubvMU6ZzvHPFC8i7whvMm7yDyHvAe7jDz4ugu6HrtQvI48rbszOcu8sLxPPMC7AbyOvKk7ozyZPIG8ibyvvCC8Mzu2vMK8YTyRu4g8TLxMPNE8sruJvMS8tLyXvHy81ztoO7e8K7xGvIk8Srx5O4K8zzyBvII8SbyMvGU7E7yeO3u80LuBPB48lzxmO3U8d7w7vLo8nTzIvAC8Frw3PFg8ULymvMQ7kLy4uzM8qbvOvJW8vbt9PLo8xTunvFM6pDwRvCo8jTugPIO8fDyXu8o8Dzypu588CDyjPGc8ELyUvIW8p7wpvIO7hryGvJm8jTvsOiQ8XTvMPNI7szt4PKe7NbsYvCI8XTy6O508LzyzPEm8/DvIu5y8pzyFvMY8ozyPu1W7wzw/vLg8uzxxPE88SLvOPHe7bryxPL88ZzyhvKM8KryyOkW8ljx2vHy8oryRPFM8mjikPBw8jrz4OnM8BLx9vE08tLuUPLI8gbmWOtq7XDxkPG48h7y0vGC8yryoPNE8jjv9u+k7gjxtPJU87LoqO/W7vbwrvCM8dLvFvI68l7yfOzE50LwlO648yTzFuqm7Ebx9vHG7ezxxvBq8WrynPNE8Cju4PFo8g7wUvJc8oLx/vKy7NTrNvHw8tbqNvDI8pLeGutE8srxkPMC8uzy0PA48XzwNPG68srzJvJu8oDzNPFy8DDxPvM88lzyrPLs7sbyhvJY8DDgpuim8/buLvBe8pzxvO428wLwjvLK8nzxnPIG8uLywvOo7jLyeO0G6uzzLvKS8jTzRvKG8mLxmvL87tbyoPJQ6MzyBPMU8sbs8vNY7irykPEY86zu5uwm8PTzLvI48MjwHO1M85zu/vLO88ru6vJk8ObxMO9C8rryUvK+7ozyeOzi8yrwDvG67+ztAvLG6Zjt7vBC8OzxFu7a8STyGPA08lTyfPIo8KDsdvM28nrykvGS7l7yFvME8ITxUPGK6xTyUPI28ejvHvIU8xbqlPCQ8frwDvLy8vrxMPJa8yDwVPLI8abycvCm8Mbu4vDa8T7x3u0g8Zzw5vNG7urx1PHu817kjO3E4n7vLPHU8yLtkvNg6ubuOPJ68I7yXOz88o7wFu648kryFPIA8brxlO0480Lw6vF+7Qzx/vBI8vzzmu446RLxRvKO8BDsLPKW8hTuGvFW8nrwbu7e8vLzaO3w8t7sjO8e8Sbs1u5I8TbyKvLU7szwvvIu8zbyAvEm8zjy1PIq7V7xjPAS8U7yAvFA8rDz8u7870LwavC68bDysPI07Rby4vKk7iTx9PJw8pjymvCc7Jrweu5S8MTsevL68XTyVvJ48OLxOPK284rt5u5+8tTwnvLQ8j7sLPKS8GjzTO3M8irwZO8877rmyvLS8lLxlO466jbwWPHO82ztzPIK8sruTPCQ8ortmPEm8TLwcPLc8ITwOvNm6zjznu068jLyNujK7WrtOPAM8rzusurK7ALt2vDa8mDwivBc8pjw3vFE8KryKPNG8Gzx2vDu8kzx4PGe69bk0uaq7irsnPOc6MTt3usE867uNPG68mjyavJk86Ls1PGs8BTytvIi80DzIOqq8lby8u7o8VDwFPKe6jDx5vE07mjuwPK28Jby0vMe8lDzSu0O8R7vJvJw8Bjy9O4w7GDzQPJs8gjzFOWG7OjwOPGi8pDzIO3+8Z7xgOr08rrySPPi7mrylPFM8ubyrPLy8pDufPI68yzxkPK+8i7zHvD4867sgPB+84LvJuz+8XbytO/Y7hbyLu3k8nbvDPMK8pLt9PDq8FDwLOac8vjswvMc8TDw9vHo8GzxSPKy8O7snPIy8dLx+PFa8kLwkO088vzzFvKO7gzz2us48z7xYure8jLy5vJK8x7y5vJ27qzuGu+47X7wcu+M7hLqlPJC8lzwaPKm8O7wTvLC8vLytvKk8ebxzuYa8KjyOvLy8lLtjO+U7zLtcvLO8VTzAuxY8xzsIvKK82rrauyI8i7yROsY8v7u6PCo85DvlOng8oDuPPGu83LtKPMO8fDxQSwcINUSjswAMAAAADAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPAEMAYXJjaGl2ZS92ZXJzaW9uRkI/AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAHgAyAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZEZCLgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMTE0NjQ0MjcyMzg0OTM1NTA0MTUwMDAwMDgxNDI2MDAzOTk3NDQ4MFBLBwjxSztLKAAAACgAAABQSwECAAAAAAgIAAAAAAAAhscqGusAAADrAAAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAACFPeMZBgAAAAYAAAARAAAAAAAAAAAAAAAAADsBAABhcmNoaXZlL2J5dGVvcmRlclBLAQIAAAAACAgAAAAAAAA1RKOzAAwAAAAMAAAOAAAAAAAAAAAAAAAAAJYBAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAABAOAABhcmNoaXZlL3ZlcnNpb25QSwECAAAAAAgIAAAAAAAA8Us7SygAAAAoAAAAHgAAAAAAAAAAAAAAAACSDgAAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkUEsGBiwAAAAAAAAAHgMtAAAAAAAAAAAABQAAAAAAAAAFAAAAAAAAAEIBAAAAAAAAOA8AAAAAAABQSwYHAAAAAHoQAAAAAAAAAQAAAFBLBQYAAAAABQAFAEIBAAA4DwAAAAA="""

def create_checkpoint():
    """创建检查点文件"""

    # 目标路径
    CKPT_DIR = "/root/复现/文章/第四章引用/引用的代码/multi-bit-text-watermark-master/ckpt"
    TARGET_FILE = os.path.join(CKPT_DIR, "WatermarkDecoder-v_head.pt")

    print("=" * 60)
    print("创建 WatermarkDecoder-v_head.pt 检查点文件")
    print("=" * 60)
    print()

    # 检查文件是否已存在
    if os.path.exists(TARGET_FILE):
        print(f"检查点文件已存在: {TARGET_FILE}")
        size_kb = os.path.getsize(TARGET_FILE) / 1024
        print(f"大小: {size_kb:.1f} KB")
        return True

    # 创建目录
    print(f"[1/2] 创建目录: {CKPT_DIR}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("     完成")
    print()

    # 解码并写入文件
    print("[2/2] 创建检查点文件...")
    try:
        data = base64.b64decode(CHECKPOINT_B64)
        with open(TARGET_FILE, 'wb') as f:
            f.write(data)

        size_kb = len(data) / 1024
        print(f"     完成")
        print(f"     路径: {TARGET_FILE}")
        print(f"     大小: {size_kb:.1f} KB")
        print()

        # 验证文件
        try:
            import torch
            state_dict = torch.load(TARGET_FILE, map_location='cpu', weights_only=True)
            print(f"     验证: PyTorch 文件正常")
            print(f"     包含 {len(state_dict)} 个张量")
        except Exception as e:
            print(f"     警告: 验证时出现问题: {e}")

        print()
        print("=" * 60)
        print("创建完成！现在可以运行 test9_multibit_watermark.py")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"     错误: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = create_checkpoint()
    sys.exit(0 if success else 1)
