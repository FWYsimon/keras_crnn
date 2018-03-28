from PIL import Image
import os
import json
import numpy as np
import random
import math

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, args):
		self.image_path = args.image_path
		self.text_path = args.text_path
		self.json_path = args.json_path
		self.save_path = args.save_path
		self.key_path = args.key_path
		self.json_val_path = args.json_val_path

		self.img_height = args.img_height
		self.img_width = args.img_width


	# 数据预处理
	def data_preprocess(self):
	    i = 1
	    dic = {}
	    # max_height, min_height, max_label_length = -1, -1, -1
	    # max_label_length_file = ""
	    # max_height_file = ""
	    for root, dirs, files in os.walk(self.image_path):  
	        for file in files:  
	            #file_name_list.append(os.path.splitext(file)[0])
	            file_name = os.path.splitext(file)[0]
	            text_file = self.text_path + file_name + ".txt"
	            raw_img = Image.open(self.image_path + file)
	            with open(self.text_file, encoding='utf-8') as f:
	                for labels in f.readlines():
	                    labels = labels.strip('\r\n')
	                    arr = labels.split(',')
	                    
	                    x1 = float(arr[0]) #左上角
	                    y1 = float(arr[1])
	                    x2 = float(arr[2]) #左下角
	                    y2 = float(arr[3])
	                    x3 = float(arr[4]) #右下角
	                    y3 = float(arr[5])
	                    x4 = float(arr[6]) #右上角
	                    y4 = float(arr[7])
	                    label = arr[8]
	                    label_length = len(label)

	                    if (label == "###"):
	                        continue
	                    
	                    left = int(x1 if x1 < x2 else x2)
	                    top = int(y1 if y1 < y4 else y4)
	                    right = math.ceil(x3 if x3 > x4 else x4)
	                    bottom = math.ceil(y2 if y2 > y3 else y3)
	                    
	                    if (left == right or top == bottom):
	                        continue
	                    temp = left
	                    if (left > right):
	                        left = right
	                        right = temp
	                    temp = top
	                    if (top > bottom):
	                        top = bottom
	                        bottom = temp
	                    box = (left,top,right,bottom)
	                    
	                    print(box)
	                    print(label_length)
	                    height = bottom - top
	                    width = right - left
	                    aspect_ratio = height / width
	                    # 剪裁图片
	                    if (width > height or aspect_ratio <= 1.5):
	                        img = horizontal_image_method(raw_img, box)
	                    else:
	                        img = vertical_image_method(raw_img, box, label_length)
	                    
	                    # # 获取最大和最小height
	                    # width, height = img.size
	                    # if (i == 1):
	                    #     min_height = height
	                    # if (max_height < height):
	                    #     max_height = height
	                    #     max_height_file = text_file
	                    # if (min_height > height):
	                    #     min_height = height
	                    # if (max_label_length < label_length):
	                    #     max_label_length = label_length
	                    #     max_label_length_file = text_file
	                    # 保存图片
	                    image_name = str(i) + ".jpg"
	                    img.save(self.save_path + image_name, quality=100)
	                    
	                    dic[image_name] = str(label)
	                
	                    i = i + 1
	    json_file = open(self.json_path ,'w', encoding='utf-8')
	    json.dump(dic, json_file, ensure_ascii=False)
	    json_file.close()
		

		# 处理横的训练集
	def horizontal_image_method(self, img, box):
	    region = img.crop(box)
	    region = region.convert('L')
	    return region

	# 处理竖的训练集
	def vertical_image_method(self, img, box, label_length):
	    region = img.crop(box)
	    region = region.convert('L')

	    # 将竖的图片转成横的图片
	    region_width, region_height = region.size
	    one_word_height = int(region_height / label_length)
	    temp_region_left = 0
	    if (one_word_height == 0):
	        return img
	    
	    new_img = Image.new('L', (region_width * label_length, one_word_height))
	    
	    for index in range(label_length):
	        temp_region_height = one_word_height * index
	        temp_region = region.crop((0, temp_region_height, region_width, temp_region_height + one_word_height))
	        new_img.paste(temp_region, (temp_region_left, 0, temp_region_left + region_width, one_word_height))
	        temp_region_left += region_width
	    return new_img

	# 缩放图片
	def rescale(self):
	    for root, dirs, files in os.walk(self.image_path):  
	        for file in files:
	            img = Image.open(self.image_path + file)
	            img = img.resize((self.img_width, self.img_height), Image.ANTIALIAS)
	            
	            img.save(self.save_path + file, quality=100)

	# 获取所有字符集
	def generate_key(self):
		char = ""
		with open(self.json_file, 'r', encoding='utf-8') as f:
	        image_label = json.load(f)
	    labels = [j for i, j in image_label.items()]
	    for label in labels:
	        label = str(label)
	        for i in label:
	            if (i in char):
	                continue
	            else:
	                char += i
	    # 所有字符长度
	    self.nclass = len(char)
		key_file = open(self.key_path, 'w', encoding='utf-8')
		key_file.write(char)
		key_file.close()

	# 随机取一部分数据为测试集
	def random_get_val(self):
		with open(self.json_path, 'r', encoding='utf-8') as f:
        	image_label = json.load(f)
	    image_file = [i for i, j in image_label.items()]
	    # 所有训练集的长度
	    nums = len(image_file)
	    resultList = random.sample(range(1, nums + 1), 3000)
	    dic = {}
	    for num in resultList:
	        image_name = str(num) + '.jpg'
	        dic[image_name] = image_label[image_name]
	        image_label.pop(image_name)
	    self.train_length = len(image_label)
	    self.val_length = len(dic)
	    json_val_file = open(self.json_val_path,'w', encoding='utf-8')
	    json.dump(dic, json_val_file, ensure_ascii=False)
	    json_val_file.close()
	    json_meta_file = open(self.json_path, 'w', encoding='utf-8')
	    json.dump(image_label, json_meta_file, ensure_ascii=False)
	    json_meta_file.close()

	# 生成一个batch的数据
	def generate(self, json_path, image_path, char_path, batch_size, max_label_length, image_size):
	    with open(json_path, 'r', encoding='utf-8') as f:
	        image_label = json.load(f)
	    f = open(char_path, 'r', encoding='utf-8')
	    char = f.read()
	    char_to_id = {j:i for i,j in enumerate(char)}
	    id_to_char = {i:j for i,j in enumerate(char)}
	    image_file = [i for i, j in image_label.items()]
	    
	    x = np.zeros((batch_size, image_size[0], image_size[1], 1), dtype=np.float)
	    labels = np.ones([batch_size, max_label_length]) * 10000
	    input_length = np.zeros([batch_size, 1])
	    label_length = np.zeros([batch_size, 1])
	    
	    r_n = random_uniform_num(len(image_file))
	    
	    image_file = np.array(image_file)
	    while 1:
	        shuffle_image = image_file[r_n.get(batch_size)]
	        for i, j in enumerate(shuffle_image):
	            img = Image.open(image_path + j)
	            img_arr = np.array(img, 'f') / 255.0 - 0.5
	            
	            x[i] = np.expand_dims(img_arr, axis=2)
	            label = image_label[j]
	            
	            label_length[i] = len(label)
	            input_length[i] = image_size[1] // 4 + 1
	            labels[i,:len(label)] = [char_to_id[i] for i in label]
	        inputs = {'the_input': x,
	                  'the_labels':labels,
	                  'input_length':input_length,
	                  'label_length':label_length
	        }
	        outputs = {'ctc':np.zeros([batch_size])}
	        yield (inputs, outputs)


	class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
	    def __init__(self,total):
	        self.total = total
	        self.range = [i for i in range(total)]
	        np.random.shuffle(self.range)
	        self.index = 0
	    def get(self,batch_size):
	        r_n=[]
	        if(self.index + batch_size > self.total):
	            r_n_1 = self.range[self.index:self.total]
	            np.random.shuffle(self.range)
	            self.index = (self.index + batch_size) - self.total
	            r_n_2 = self.range[0:self.index]
	            r_n.extend(r_n_1)
	            r_n.extend(r_n_2)
	            
	        else:
	            r_n = self.range[self.index:self.index + batch_size]
	            self.index = self.index + batch_size
	        return r_n  