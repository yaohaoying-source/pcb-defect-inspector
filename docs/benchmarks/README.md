# PCB 300 图批测数据口径

`pcb-300-summary.json` 是从私有备份聚合生成的公开摘要，不包含原始图像、文件名、个人路径、逐条结果或服务器地址。

| 摘要字段 | 备份中的来源 |
| --- | --- |
| `dataset_split` | `userdata/Model/pcb_test_300/labels.csv` 的 split 字段统计 |
| 图像数量、各类别数量、阈值、命中、耗时与吞吐 | `userdata/Model/pcb_test_300/results/run_20260617_083927/summary.json`。命中和无检测数与 `batch_detection_result.csv` 交叉核对 |
| `monitor` | 同一次运行的 `system_monitor.csv`。峰值和最低值均为离散采样统计 |
| `http_upload` / `tcp_upload` | 同一次运行的 `upload_result.csv` / `upload_tcp_result.csv`。TCP 成功是客户端函数返回记录 |

统计脚本按图片中预测的类别是否包含期望类别计数，未对预测框与真值框做 IoU 匹配。数据集来源包含 train、val、test，不适合把结果称作独立测试集上的检测准确率或 mAP。
