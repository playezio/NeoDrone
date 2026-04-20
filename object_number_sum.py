import pandas as pd
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='统计各组目标数量')
parser.add_argument('--input', type=str, default='xxx.xlsx', help='输入Excel文件路径')
args = parser.parse_args()

file_path = args.input
df = pd.read_excel(file_path, header=None)

groups = []
current_group = None
image_count = 0
target_total = 0
in_header = False

for idx, row in df.iterrows():
    val = str(row[0]).strip()

    if val.startswith("图像集合计"):
        current_group = val
        image_count = 0
        target_total = 0
        in_header = True  # 下一行是表头，跳过去

    elif in_header and val == "图像数量":
        # 这是表头行，跳过
        in_header = False
        continue

    elif val == "图像数量":
        # 真正的图像数量行
        image_count = int(row[1]) if pd.notna(row[1]) else 0

    elif val == "合计":
        target_total = int(row[2]) if pd.notna(row[2]) else 0
        groups.append({
            "组名": current_group,
            "图像数量": image_count,
            "目标数量合计": target_total
        })

# 输出结果
result_df = pd.DataFrame(groups)
print("各组明细：")
print(result_df)
print("\n总图像数量：", result_df["图像数量"].sum())
print("总目标数量：", result_df["目标数量合计"].sum())