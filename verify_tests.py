#!/usr/bin/env python3
"""验证数据接入层测试通过情况"""
import subprocess
import sys

# 运行测试
result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/test_data_provider.py', '-v', '--tb=no'],
    capture_output=True,
    text=True
)

# 提取测试结果
lines = result.stdout.split('\n')
passed_count = 0
failed_count = 0
test_results = []

for line in lines:
    if 'PASSED' in line:
        passed_count += 1
        # 提取测试名称
        test_name = line.split('::')[1] if '::' in line else 'unknown'
        test_results.append(f"✓ {test_name}")
    elif 'FAILED' in line:
        failed_count += 1
        test_name = line.split('::')[1] if '::' in line else 'unknown'
        test_results.append(f"✗ {test_name}")

# 显示摘要
print("=" * 60)
print("测试执行结果")
print("=" * 60)
print(f"\n通过: {passed_count} ✓")
print(f"失败: {failed_count} ✗")
print(f"总计: {passed_count + failed_count}")
print(f"\n通过率: {100 * passed_count / (passed_count + failed_count):.1f}%")

if failed_count > 0:
    print("\n失败的测试:")
    for result in test_results:
        if result.startswith('✗'):
            print(f"  {result}")

# 显示最后的摘要行
for line in lines[-10:]:
    if 'passed' in line or 'failed' in line:
        print(f"\n{line}")
        break

print("\n" + "=" * 60)
sys.exit(0 if failed_count == 0 else 1)
