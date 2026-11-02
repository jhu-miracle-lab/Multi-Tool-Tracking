# FPRT 入口

`config.py` 显式设置 `ALLOW_PRIMARY_SWITCH = False`。三个启动脚本调用根目录的共享实现，不再通过符号链接切换算法配置。

从 `project_code/` 执行：

```bash
python FPRT/offline_tracking.py --data /path/to/dataset --no_display
python FPRT/ros_tracking.py
python FPRT/ros_offline.py --ros-args -p data:=/path/to/dataset
```

配置、算法与接口说明见 [共享项目说明](../README.md)。
