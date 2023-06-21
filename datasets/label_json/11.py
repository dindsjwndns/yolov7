# trans_others_to_jpg.py

import os
import cv2 as cv

image_path = 'D:/360驱动大师目录/2.png'  # 设置图片读取路径
save_path = 'C:/'  # 设置图片保存路径，新建文件夹，不然其他格式会依然存在

if not os.path.exists(save_path):  # 判断路径是否正确，并打开
    os.makedirs(save_path)

image_file = os.listdir(image_path)
# print(image_file)
for image in image_file:
    # print(image)
    if image.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png', 'JPG', 'PNG']:
        str = image.rsplit(".", 1)  # 从右侧判断是否有符号“.”，并对image的名称做一次分割。如112345.jpeg分割后的str为["112345","jpeg"]
        # print(str)
        output_img_name = str[0] + ".png"  # 取列表中的第一个字符串与“.jpg”放在一起。
        # print(output_img_name)
        dir = os.path.join(image_path, image)
        # print("dir:",dir)
        src = cv.imread(dir)
        # print(src)
        cv.imwrite(save_path + output_img_name, src)
print('FINISHED')
