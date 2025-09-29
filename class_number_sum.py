import pandas as pd
from collections import defaultdict

# 读取 Excel 文件
file_path = "xxx.xlsx"
df = pd.read_excel(file_path, header=None)

category_totals = defaultdict(int)
current_group = None
in_data = False

for idx, row in df.iterrows():
    val = str(row[0]).strip()

    if val.startswith("图像集合计"):
        current_group = val
        in_data = False
    elif val == "图像数量":
        in_data = True
        continue
    elif val == "合计":
        in_data = False
    elif in_data:
        try:
            category = int(row[1])
            count = int(row[2])
            category_totals[category] += count
        except (ValueError, TypeError):
            continue

# 生成结果表
result = pd.DataFrame(list(category_totals.items()), columns=["类别", "总数量"])
result = result.sort_values("类别").reset_index(drop=True)

# 保存为 Excel
output_path = "类别统计汇总.xlsx"
result.to_excel(output_path, index=False)

print("✅ 已生成汇总表：", output_path)
print(result)
print("\n总计类别数：", len(result))
print("总计目标数量：", result["总数量"].sum())