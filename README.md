# Multi-Tool Tracking：共享实现与两个入口

FPRT 和 APRT 共用一套算法代码，入口显式传入 `allow_primary_switch`。选择方法不再依赖符号链接、导入路径或全局配置。

```text
project_code/
├── tracking/                 # 共享跟踪实现
│   ├── frame.py              # ArmObservation、ArmFrame
│   ├── arm_tracker.py        # 单臂几何、观测、预热、位姿与绘图
│   ├── primary_selector.py   # 独立的可靠度迟滞状态机
│   ├── combined_tracker.py   # 共享 PF、粒子重锚定与累计相对更新
│   ├── offline_session.py    # 数据加载、预热、绘图及状态保存
│   └── offline_tracking.py   # 共用命令行流程，直接运行默认 FPRT
├── ros2_node/                # ROS 适配层
├── FPRT/                     # config.py 与三个启动脚本
├── APRT/                     # config.py 与三个启动脚本
├── core/                     # 原有 PF/机器人基础实现，内容未改
├── Parameter/、models/       # 共用标定、几何与检测模型资源
├── keypoints/、edge_detection/ # 数据准备及检测辅助脚本
├── launch/                   # 共用 ROS launch 文件
└── setup.py                  # 从此目录构建 ROS 包
```

## 启动

在已配置依赖的 tracking Python 环境中，从 `project_code/` 执行：

```bash
python FPRT/offline_tracking.py --data /path/to/dataset --no_display
python APRT/offline_tracking.py --data /path/to/dataset --no_display
```

两个入口都支持 `--handeye`、`--n_particles`、`--out_dir`、`--save_states`。默认输出分别位于 `FPRT/tracking_output_offline/` 和 `APRT/tracking_output_offline/`。

从代码创建时显式选择方法；两种方法可同时存在于同一 Python 进程：

```python
fixed = CombinedArmTracker(arms, 500, allow_primary_switch=False)
auto = CombinedArmTracker(other_arms, 500, allow_primary_switch=True)
```

实例应使用各自的 ArmTracker 对象，因为观测及位姿状态是可变的。

## 数据和职责边界

`ArmTracker.prepare_frame()` 返回 `ArmFrame`；`CombinedArmTracker.update()` 返回 `{arm_name: ArmFrame}`。缺少关节数据时 `frame.available` 为 False；没有视觉检测但关节数据可用时仍是一帧有效数据。

```python
frames = combined.update(frame_idx, left_img=image)
frame = frames['psm1']
if frame.available:
    tracker.draw(image, frame.corrected_keypoints)
    count = frame.observation_count
```

`ArmObservation` 保存关键点、三维点、杆几何、杆边缘和图像尺寸。观测重分配修改其中的 `keypoints`；`observation_count` 保留重分配前的数量，保持原输出含义。

单臂公开接口：`anchor`、`has_relative_anchor`、`set_anchor()`、`warmup_std`、`shaft_radius`、`log_likelihood()`、`finalize_pose()`。`set_anchor()` 只改变几何参考；共享粒子的坐标转换由 CombinedArmTracker 负责。跨臂模块不再直接写单臂私有状态。

PrimarySelector 只维护主臂候选、连续确认次数和驻留时间，不依赖图像、机器人几何或 PF。CombinedArmTracker 根据其返回的主臂执行重锚定与粒子更新。

共享 PF 内部仍调用原 `core/ParticleFilter.py` 接口；本次没有修改 core，也没有改变粒子更新顺序或似然公式。

## ROS

从本目录重新构建并 source 新安装环境。旧 `FPRT/build`、`install`、`log` 已归档至 `../test_code/legacy_ros_build/FPRT/`，不作为新版运行环境。

在 ROS 环境中可直接启动对应脚本：

```bash
python FPRT/ros_tracking.py
python APRT/ros_tracking.py
python APRT/ros_offline.py --ros-args -p data:=/path/to/dataset -p no_display:=true
```

安装后的 console entries 包含 `fprt_tracking_node`、`aprt_tracking_node`、`fprt_offline_node`、`aprt_offline_node`、`fprt_offline`、`aprt_offline`。原 `tracking_node` / `offline_node` 继续可用，默认 FPRT；通过 ROS 参数 `allow_primary_switch:=true` 选择 APRT。两个 launch 文件也接受该参数。

## 方法与迁移

两种入口均使用可靠度门控、关键点重分配、2D 杆观测、重叠降权、模式切换重初始化和累计主臂误差传递。FPRT 只允许第一臂为主臂；APRT 可以选择任一臂。

累计变化为 `G = L_current @ inverse(L_reference)`；参考在进入相对模式或切换主臂时建立。副臂保留自身六维误差状态。

- 旧 `FPRT/tracking/`、`APRT/tracking/` 已由根目录 `tracking/` 取代。
- 方法配置在 `FPRT/config.py`、`APRT/config.py`；共享数值常量在 `tracking/tracker_config.py`。
- 原结果元组改为 ArmFrame，外部调用脚本需要按上述字段取值。
- NPZ 的 `frames`、`ema_states`、`fk_tips` 字段保持不变；`ema_states` 存的是 PF 状态，沿用旧名以兼容读取脚本。
- 当前 FPRT 也有可靠度门控，与论文中始终处于相对模式的配置有区别。此次为架构重构，未重新评估完整实验精度。

建议阅读顺序：`frame.py` → `offline_session.py` → `arm_tracker.py` → `primary_selector.py` → `combined_tracker.py` → 两类 observation 模块。
