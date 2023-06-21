# change_json_imagePath.py

import json
import os
import re

path = 'E:\pythonProject1\yolov7\datasets\label_json'  # json文件路径
dirs = os.listdir(path)

num_flag = 0
for file in dirs:  # 循环读取路径下的文件并筛选输出
    if os.path.splitext(file)[1] == ".json":  # 筛选Json文件
        num_flag = num_flag + 1
        print("path = ", file)  # 此处file为json文件名，之前修改为与图片jpg同名
        # print(os.path.join(path,file))
        with open(os.path.join(path, file), 'r') as load_f:  # 若有中文，可将r改为rb
            load_dict = json.load(load_f)  # 用json.load()函数读取文件句柄，可以直接读取到这个文件中的所有内容，并且读取的结果返回为python的dict对象
        n = len(load_dict)  # 获取字典load_dict中list值
        print('n = ', n)
        print("imagePath = ",
              load_dict['imagePath'])  # 此处因为我的json文件要修改的imagePath， 没有那么多弯弯绕， 直接在顶层， 所以一层[]即可， 如果你们的不是这种结构， 需自行修改

        filename = file[:-5]  # 去掉拓展名5位  .json
        print("filename = ", filename)
        load_dict['imagePath'] = filename + '.jpg'  # 存到当前路径下， 如果有其它存储要求， 自行修改即可
        print("new imagePath = ", load_dict['imagePath'])

        with open(os.path.join(path, file), 'w') as dump_f:
            json.dump(load_dict, dump_f)

if (num_flag == 0):
    print('所选文件夹不存在json文件，请重新确认要选择的文件夹')
else:
    print('共{}个json文件'.format(num_flag))
