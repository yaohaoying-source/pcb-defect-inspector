# 模型权重清单

权重未包含在源码仓库。以下 SHA256 来自 ELF 2 的私有备份归档 `01-PCB部署与数据.tar.gz`，供发布前核对。相同文件名可能代表不同模型，请保持区分。

| 建议 Release 文件名 | 备份归档内路径 | 大小（字节） | SHA256 |
| --- | --- | ---: | --- |
| `PCB_Improve.rknn` | `userdata/PCB_Improve.rknn` | 5,107,464 | `a75166625d0ce2bc2eefb19c82bb7cd355ad0df0e9e28fb4e9b81d0c1549c1fa` |
| `yolo11-app.rknn` | `userdata/yolo11.rknn` | 4,247,819 | `d744606ec52ac992cb91187a109c57c42dad9fc21b625f22e9c250cb69a83a25` |
| `yolo11-model-dir.rknn` | `userdata/Model/yolo11.rknn` | 4,206,536 | `b3b61b677de378c4f52c9ee5d3ff9bccc7356094457752653cb9324d53e76732` |
| `yolo11_pcb.rknn` | `userdata/yolo11_pcb.rknn` | 4,883,280 | `57495d00c2630ea0104448ae9a7990b1beefdf07d430c6cf5a5617beba96074f` |

批测记录与备份的旧版 Qt 设置 `detection/last_model` 均指向文件名 `PCB_Improve.rknn`。主应用的 `ui/detection_tab.py` 和 `ui/camera_tab.py` 通过文件选择器加载用户选定的模型，并保存最近选择的路径，并未固定加载某一份 `yolo11.rknn`。因此，第一项是复现现有批测的已确认权重；其余三项是待确认训练来源、类别和授权后再决定是否发布的候选。

`userdata/Model/PCB_Improve.rknn` 与第一项完全相同。板端部署目录里的 `yolo11.rknn` 与第二项完全相同。上表中的两个 `yolo11` 文件不同，尚需确认各自的训练来源、类别和使用场景。批测记录明确使用第一项，其余权重未在该次批测中验证。

在板端放入 `pcb_defect_inspector/models/` 后，可用 `sha256sum pcb_defect_inspector/models/*.rknn` 核对。发布前由维护者核实权重与训练数据的来源、使用许可和对应的模型转换工具链。若不打算发布无法确认授权的权重，可改为在 Release 中提供合规获取与自行转换的方法。
