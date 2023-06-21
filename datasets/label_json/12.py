# rename.py

import os

path = "E:\pythonProject1\yolov7\datasets\IMA"  # json标签文件的保存路径
filelist = os.listdir(path)
count = 1
for file in filelist:
    print(file)
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    Newdir = os.path.join(path, str(count).zfill(6) + filetype)  # zfill(6):表示命名为6位数
    os.rename(Olddir, Newdir)

    count += 1
