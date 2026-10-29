"""随机删除训练集中的图片及其标签，把数据集裁剪到指定数量。

注意：此脚本会永久删除文件，执行前请确认已备份数据集。

用法（在仓库根目录执行）：
    python training/tools/delete.py --target 100
    python training/tools/delete.py --target 100 --dry-run
"""

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DIR = ROOT / "training" / "dataset" / "train"


def delete_random_files(train_dir: Path, target_count: int, dry_run: bool = False):
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"

    image_files = [
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    current_count = len(image_files)
    if current_count <= target_count:
        print("当前文件数量已经小于或等于目标数量，无需删除。")
        return

    delete_count = current_count - target_count
    print(f"当前文件数量: {current_count}, 需要删除: {delete_count}")

    for file_name in random.sample(image_files, delete_count):
        image_path = images_dir / file_name
        label_path = labels_dir / f"{image_path.stem}.txt"

        if dry_run:
            print(f"[dry-run] 将删除: {image_path}")
            if label_path.exists():
                print(f"[dry-run] 将删除: {label_path}")
            continue

        if image_path.exists():
            image_path.unlink()
        if label_path.exists():
            label_path.unlink()

    if not dry_run:
        print(f"删除完成，剩余图片: {len(list(images_dir.iterdir()))}")


def main():
    parser = argparse.ArgumentParser(description="随机裁剪训练集图片与标签")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="训练集目录")
    parser.add_argument("--target", type=int, default=100, help="目标图片数量")
    parser.add_argument("--dry-run", action="store_true", help="只打印不删除")
    args = parser.parse_args()

    delete_random_files(Path(args.train_dir), args.target, args.dry_run)


if __name__ == "__main__":
    main()
