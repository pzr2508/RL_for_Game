import os
import cv2

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def resize_longest_edge_cv2(img, max_size=640):
    h, w = img.shape[:2]
    longest = max(w, h)
    scale = max_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_images_opencv(root_dir, max_size=640):
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue

            img_path = os.path.join(root, fname)

            img = cv2.imread(img_path)
            if img is None:
                print(f"✗ 读取失败: {img_path}")
                continue

            resized = resize_longest_edge_cv2(img, max_size)

            # 如果尺寸没变，可以直接跳过写盘（更快）
            if resized.shape == img.shape:
                continue

            cv2.imwrite(img_path, resized)
            print(f"✓ {img_path}")


if __name__ == "__main__":
    resize_images_opencv("./train_data/saved_videos_tmp", 640)
