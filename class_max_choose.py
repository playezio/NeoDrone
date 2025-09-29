import pandas as pd
from collections import defaultdict
import os

file_path = "xxx.xlsx"
df = pd.read_excel(file_path, header=None)

# 收集：类别 -> [(组名, 数量), ...]
category_groups = defaultdict(list)
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
            category_groups[category].append((current_group, count))
        except (ValueError, TypeError):
            continue

# 对每个类别取 Top1
top1_rows = []
for cat, grp_list in category_groups.items():
    # 按数量降序
    grp_list.sort(key=lambda x: x[1], reverse=True)
    top_group, top_count = grp_list[0]
    top1_rows.append({"类别": cat, "最多组名": top_group, "该组数量": top_count})

# 生成 DataFrame 并排序
result = pd.DataFrame(top1_rows).sort_values("类别").reset_index(drop=True)

# 保存
out_file = "各类别Top1组.xlsx"
result.to_excel(out_file, index=False)
print("✅ 已生成：", os.path.abspath(out_file))
print(result)