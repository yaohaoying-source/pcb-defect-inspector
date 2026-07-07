# Project Structure

```text
pcb-defect-inspector-open-source/
  README.md
  .gitignore
  docs/
    PROJECT_STRUCTURE.md
  pcb_defect_inspector/
    main.py
    start_pcb_inspector.sh
    requirements.txt
    assets/
      app_background.jpg
      app_background_faded.jpg
      pcb_defect_icon.png
    deepseek/
      api_client.py
    detection/
      base_detector.py
      postprocess.py
      rknn_detector.py
      tracker.py
    models/
      .gitkeep
    outputs/
      .gitkeep
    ui/
      camera_tab.py
      chat_tab.py
      detection_tab.py
      main_window.py
      settings_dialog.py
    upload/
      protocol.py
      uploader.py
    utils/
      config.py
      image_loader.py
      model_downloader.py
      result_paths.py
    vl/
      qwen_prompt.py
      qwen_vl_client.py
```

## 核心模块说明

- `ui/camera_tab.py`: 实时摄像头检测线程，GStreamer/V4L2 采集，Qt 显示，截图和上传入口。
- `ui/detection_tab.py`: 图片检测界面，加载图片后调用检测器并展示结果。
- `detection/rknn_detector.py`: RKNN Runtime 初始化、模型推理、检测接口。
- `detection/postprocess.py`: YOLO 输出解析、DFL、NMS 等后处理逻辑。
- `upload/uploader.py`: 板端向服务器上传截图或检测结果。
- `utils/model_downloader.py`: 从服务器 Web API 拉取 `.rknn` 模型到板端。
- `vl/qwen_vl_client.py`: 调用本地 Qwen3-VL demo 进行图片问答。
