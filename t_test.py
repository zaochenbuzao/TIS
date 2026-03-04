try:
    # 在 Python 中检查可用的数据集类
    import datasets
    print(dir(datasets))
    print("Successfully imported TIS from tis")
except ImportError as e:
    print(f"Import failed: {e}")