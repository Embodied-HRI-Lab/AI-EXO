# 下楼 solver 2x3 thigh-IMU Exo（h48@30 Hz / 100 Hz 控制器）

这里是三种彼此独立的固定轨迹求解教师所蒸馏出的下楼 Exo 候选：替代髋力矩、
activation²、weighted activation²。它们没有经过联合 RL，也没有把 Human 策略打包进来。

共同部署契约：standing-zeroed thigh-world IMU 四维输入、h48（1.6 s）、左右独立输出、
无硬反对称、峰值 +/-4.2 Nm、模型内部强制 0.21 Nm/10 ms、最新 3 个训练帧静止门。
100 Hz 包装器保留 158 帧稠密缓冲区，再按原 30 Hz 时间偏移抽取 48 帧；不会把历史窗
错误缩短成 0.48 s。

| 模型 | 留出蒸馏 RMSE | 状态 |
|---|---:|---|
| `stairdown_hip_torque_replacement_thighimu_h48_30hz_at_100hz_lzn.pt` | 1.303 Nm | 仅离线打包通过 |
| `stairdown_activation2_thighimu_h48_30hz_at_100hz_lzn.pt` | 1.308 Nm | 仅离线打包通过 |
| `stairdown_weighted_activation2_thighimu_h48_30hz_at_100hz_lzn.pt` | 1.228 Nm | 仅离线打包通过 |

下楼三份模型的留出误差约 1.23–1.31 Nm，明显高于下坡模型；因此这里的
“部署”只表示控制器可加载、输入/左右顺序/限幅/slew/静止门已验证，不能写成
闭环或实机已经可用。当前也没有满足 thigh-world 语义的真人下楼数据可回放。

在 `Higher_Controller/NN_Controller` 下先 dry-run（替换 `MODEL`）：

```bash
python tcn_controller_imu_and_torque_lowpass.py \
  --model models/202609011612_stairdown_solver_2x3_thighimu_h48_full4p2/MODEL \
  --mock-imu --dry-run --duration 10 --no-output-filter
```

实机命令仍必须显式选择模型；第一次测试保留默认 5 Hz 输出低通，并显式限制：

```bash
python tcn_controller_imu_and_torque_lowpass.py \
  --model models/202609011612_stairdown_solver_2x3_thighimu_h48_full4p2/MODEL \
  --max-torque 4.2 --arm
```

上人前必须先核对左右 IMU、站立归零、屈髋正方向、电机方向、急停和电流档。
控制器默认 10 Hz IMU 低通；源仿真输入为直接 30 Hz 采样，这一差异尚需实机 A/B。
