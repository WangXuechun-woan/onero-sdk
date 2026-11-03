"""
错误处理与安全恢复 — 小白必读。

演示 SDK 在各种错误场景下的行为，以及如何安全恢复。
所有错误都抛出 OneroError（不会静默失败），用户只需 try/except。

运行 (仿真):
    python examples/20_error_handling.py
"""

from onero_sdk import Arm, OneroError
from onero_sdk.mujoco_backend import MujocoBackend

backend = MujocoBackend(model_name="A1R")
arm = Arm("A1R", backend=backend)

# === 1. 未使能就调用 ===
print("1. 未使能就移动:")
try:
    arm.movej([0.0] * 7)
except OneroError as e:
    print(f"   ✓ 捕获错误: {e}")

# === 2. 使能后正常操作 ===
arm.enable()
print("\n2. 正常移动:")
arm.movej([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0)
print("   ✓ 成功")

# === 3. 参数数量错误 ===
print("\n3. 关节数量错误:")
try:
    arm.movej([0.0, 0.5, 0.3])  # 只给了 3 个，需要 7 个
except OneroError as e:
    print(f"   ✓ 捕获错误: {e}")

# === 4. 参数类型错误 ===
print("\n4. 参数类型错误:")
try:
    arm.movej("invalid")  # 字符串不行
except OneroError as e:
    print(f"   ✓ 捕获错误: {e}")

# === 5. 超出关节限位（SDK 会自动 clamp，不会报错） ===
print("\n5. 超出关节限位:")
arm.movej([0.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0)
state = arm.state
print(f"   目标 99.0 rad，实际到达: {state.joints[1]:.3f} rad")
print("   ✓ SDK 自动限位保护，不会撞机械限位")

# === 6. 通用错误处理模式 ===
print("\n6. 推荐的错误处理模式:")
print("""
from onero_sdk import Arm, OneroError

arm = Arm("A1R", channel="can0")

try:
    arm.enable()
    arm.movej([...], duration=2.0)
    arm.movel([0.3, 0.0, 0.2], duration=2.0)
except OneroError as e:
    print(f"机械臂错误: {e}")
finally:
    # finally 确保始终释放硬件
    arm.disable()
""")

# === 7. 更简洁: 使用 with 语句 ===
print("7. 最简洁的写法 (with 语句自动 enable/disable):")
print("""
from onero_sdk import Arm

with Arm("A1R", channel="can0") as arm:
    arm.movej([...], duration=2.0)
    # 退出 with 时自动 disable，即使出错也不例外
""")

arm.disable()
backend.close()
print("\n全部演示完成!")
