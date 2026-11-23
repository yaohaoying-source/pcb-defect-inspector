# OV13855 相机调参记录

备份中 `/etc/iqfiles/ov13855_CMK-OT2016-FV1_default.json` 与板端相机软件包的原文件不同，并有两份带 `bak_fps15` 后缀的历史副本。该 IQ 文件用于 Rockchip rkaiq 相机处理管线。它来自板端厂商环境，仓库不分发原始配置。

复现前确认相机型号、固件、驱动和 rkaiq 版本与备份环境一致。用 `v4l2-ctl --list-devices` 和 `systemctl status rkaiq_3A.service` 检查设备与服务。保留新系统的原配置，并由使用者手动从私有备份对照目标文件和必要的版本差异。调参效果与实际帧率需在设备上验证。
