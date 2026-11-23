# 板端备份与公开源码的差异

当前源码基于已有的公开版。对 `home/elf/face_detect_app_v2/new/face_detect_app_v2/` 的同名源码逐文件对比时，除编码 BOM、行尾差异外，明确的逻辑差异集中在设置对话框和配置类：

| 文件 | 板端差异 | 本次处理 |
| --- | --- | --- |
| `ui/settings_dialog.py` | 板端显示了具体服务器地址的提示文本 | 保留回环地址提示；增加由用户确认的旧版设置导入按钮 |
| `utils/config.py` | 板端使用旧 Qt 设置命名空间及具体服务器默认地址 | 保留公开版命名空间和回环默认地址；只在用户确认后导入新设置中缺失的旧字段，不覆盖新值；模型默认路径改为项目目录 |
| `userdata/yolo_app/pcb_detect.py` | 独立 GStreamer / V4L2 原型，集成摄像头采集、NPU 推理和绘制，部分机制已在主应用拆分实现 | 收录为 `scripts/pcb_detect.py`。运行模型与设备改为显式参数，不从该原型覆盖主应用实现 |

`pcb_defect_inspector/models/.gitkeep` 和 `outputs/.gitkeep` 继续保留。备份目录不是 Git 工作树，本记录不推断板端修改的提交状态。旧 Qt 设置可能包含密钥与真实服务器地址，迁移只操作用户自己的本机设置，不写入源码。
