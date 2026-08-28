# 坡道 thigh-IMU Exo 候选（30 Hz 网络 / 100 Hz 控制器）

这些文件可由 `tcn_controller_imu_and_torque_lowpass.py` 加载，但目前只完成
离线蒸馏与 TorchScript 等价/限幅/slew 检查，尚未完成 thigh-IMU 闭环和实机验收。

网络仍按 h8@30 Hz 训练。控制器以 100 Hz 调用时，包装器从稠密 IMU 缓冲区
抽取 30 Hz 时间间隔的 8 帧，因此每 10 ms 都有新决策，同时不把历史窗缩短。
坡道保留左右独立/common-mode 输出，不做硬反对称。

| 槽位 | 峰值 | 留出 RMSE | 源状态 |
|---|---:|---:|---|
| 上坡 activation² | +/-4.2 Nm | 0.451 Nm | gait/deployment passed; endpoint needs larger acceptance |
| 上坡 weighted activation² | +/-4.2 Nm | 0.371 Nm | gait/deployment passed |
| 上坡冲击槽位（activation² 初值） | +/-4.2 Nm | 0.517 Nm | not an independently accepted impact policy |
| 下坡 activation² | +/-2.1 Nm | 0.419 Nm | gait timing gate not accepted |
| 下坡 weighted activation² | +/-2.1 Nm | 0.475 Nm | gait timing gate not accepted |
| 下坡冲击 t81920 | +/-2.1 Nm | 0.403 Nm | paired impact diagnostic passed; gait timing gate not accepted |

建议先用 dry-run（每个模型替换 `MODEL`）：

```bash
python tcn_controller_imu_and_torque_lowpass.py --model MODEL --mock-imu --dry-run --duration 10 --no-output-filter
```

训练输入已匹配控制器默认 10 Hz IMU 低通。`--no-output-filter` 只关闭额外的
5 Hz 力矩低通；模型内部仍强制 21 Nm/s（0.21 Nm/10 ms）slew。第一次上人
必须另加较低的硬件力矩/电流档，并先核对左右 IMU 符号。

上坡冲击槽位不是独立冲击专家，而是冲击实验使用的 activation² 初值。
下坡三个源候选的 gait 时序均未正式验收；不得把“控制器可加载”写成实机可用。
