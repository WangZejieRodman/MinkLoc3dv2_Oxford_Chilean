"""快速验证Chilean点云加载"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pickle
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader

# 路径配置
BASE_PATH = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
TRAIN_PICKLE = os.path.join(os.path.dirname(__file__), "training_queries_chilean.pickle")

print("=" * 60)
print("Chilean点云加载验证")
print("=" * 60)

# 加载训练查询字典
print(f"\n加载pickle: {TRAIN_PICKLE}")
with open(TRAIN_PICKLE, 'rb') as f:
    queries = pickle.load(f)

print(f"✓ 查询数量: {len(queries)}")

# 测试加载前3个点云
pc_loader = PNVPointCloudLoader()
test_count = min(3, len(queries))

print(f"\n测试加载前{test_count}个点云:")
for i in range(test_count):
    query = queries[i]
    rel_path = query.rel_scan_filepath
    full_path = os.path.join(BASE_PATH, rel_path)

    print(f"\n[{i + 1}] 文件: {rel_path}")
    print(f"    完整路径: {full_path}")
    print(f"    位置: northing={query.position[0]:.2f}, easting={query.position[1]:.2f}")
    print(f"    正样本数: {len(query.positives)}")

    try:
        # 加载点云
        pc = pc_loader(full_path)
        print(f"    ✓ 点云shape: {pc.shape}")
        print(f"    ✓ 数据类型: {pc.dtype}")
        print(f"    ✓ 点数: {pc.shape[0]}")
        print(f"    ✓ 范围: X[{pc[:, 0].min():.2f}, {pc[:, 0].max():.2f}] "
              f"Y[{pc[:, 1].min():.2f}, {pc[:, 1].max():.2f}] "
              f"Z[{pc[:, 2].min():.2f}, {pc[:, 2].max():.2f}]")
    except Exception as e:
        print(f"    ✗ 加载失败: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！Chilean点云加载正常")
print("=" * 60)