# PCB Defect Inspector

RK3588 板端 PCB 缺陷检测上位机核心源码。

## 功能概览

- PyQt 图形界面
- 图片检测与实时摄像头检测
- RKNN 模型加载和推理
- YOLO 后处理、NMS、检测框绘制
- 截图、自动截图、检测结果上传
- 从服务器拉取 `.rknn` 模型到板端 `/userdata/Model`
- 本地 Qwen3-VL/DeepSeek 辅助分析入口

## 目录结构

```text
pcb_defect_inspector/
  main.py                  # 应用入口
  ui/                      # PyQt 界面和实时检测线程
  detection/               # RKNN 推理、YOLO 后处理、跟踪器
  upload/                  # 板端上传协议和客户端
  utils/                   # 配置、图片加载、模型下载等工具
  vl/                      # Qwen3-VL 本地调用封装
  deepseek/                # DeepSeek API 客户端
  assets/                  # 图标和界面背景资源
  models/                  # 模型占位目录，实际模型不提交 Git
  outputs/                 # 运行输出占位目录，实际结果不提交 Git
```

更详细的树形说明见 `docs/PROJECT_STRUCTURE.md`。

## 运行方式

在 RK3588 板端进入源码目录：

```bash
cd pcb_defect_inspector
python3 main.py
```

或使用启动脚本：

```bash
./start_pcb_inspector.sh
```

## 模型文件

模型文件体积较大且通常不适合直接提交 Git。本仓库默认忽略：

```text
*.rknn
*.rkllm
*.onnx
*.pt
*.pth
```

板端推荐模型目录：

```text
/userdata/Model
```
