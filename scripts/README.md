# 脚本

| 脚本 | 用途 |
| --- | --- |
| `webcam_demo.py` | 不启动 Django，直接用 OpenCV 调用摄像头做表情检测（按 q 退出） |
| `generate_logo.py` | 生成项目 Logo，默认输出到本目录 `logo.png` |

```bash
python scripts/webcam_demo.py
python scripts/generate_logo.py
```

两个脚本都默认读取仓库根目录的 `weights/best.pt`。
