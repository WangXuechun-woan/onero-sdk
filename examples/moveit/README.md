# MoveIt2 示例

使用 MoveIt2 运动规划框架控制 Onero A1。MoveIt 提供碰撞检测、逆运动学和轨迹优化。

## 前提

```bash
# 启动 MoveIt (包含机械臂节点)
ros2 launch onero_moveit_config move_group.launch.py
```

## 示例列表

| # | 文件 | 说明 | 适用场景 |
|---|------|------|----------|
| 1 | `01_moveit_basic.py` | 关节空间规划与执行 | 入门、point-to-point |
| 2 | `02_moveit_cartesian.py` | 笛卡尔直线/圆弧规划 | 焊接、涂胶、切割 |
| 3 | `03_moveit_pick_place.py` | 抓取放置完整流程 | 分拣、装配 |

## MoveIt 规划组

```yaml
arm:
  joints: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
  end_effector_link: tool0
  named_states:
    home: [0, -0.5, 0, -1.0, 0, 0.5, 0]
    zero: [0, 0, 0, 0, 0, 0, 0]
```

## 常见问题

**Q: 规划总是失败?**
- 检查目标是否在工作空间内
- 检查是否与碰撞模型冲突
- 尝试增大规划时间: `arm.set_planning_time(10.0)`

**Q: 笛卡尔路径覆盖率低?**
- 减小 max_step (更密集插值)
- 检查路径中间是否有奇异点
- 尝试调整末端姿态
