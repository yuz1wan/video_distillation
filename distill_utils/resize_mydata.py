import cv2
import os

# 设置resize和crop的目标大小
target_size = (160, 120)
crop_size = (112, 112)

# 指定图片路径
img_dir = "path_to_mydata_images"
save_dir = "path_to_save_resized_images"

# 遍历图片路径下的所有图片
for subdir in os.listdir(img_dir):
    subdir_path = os.path.join(img_dir, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 读取图片
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path)

                # resize图片
                img_resized = cv2.resize(img, target_size)

                # crop图片
                h, w, _ = img_resized.shape
                top = (h - crop_size[0]) // 2
                left = (w - crop_size[1]) // 2
                img_cropped = img_resized[top:top+crop_size[0], left:left+crop_size[1]]

                # 保存处理后的图片
                save_subdir = os.path.join(save_dir, subdir)
                os.makedirs(save_subdir, exist_ok=True)
                cv2.imwrite(os.path.join(save_subdir, filename), img_cropped)
    print("Finish processing {}".format(subdir_path))

# 打印处理完成
print("All images processed!")