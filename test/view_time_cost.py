#!/user/bin/env python
#coding:utf-8
import matplotlib.pyplot as plt
import math
import numpy as np
 
 
def read_text_data(filename):
	data = []
	with open(filename, 'r') as f:  #with语句自动调用close()方法
		text = f.readline()         #跳过第一行
		line = f.readline()
		while line:
			eachline = line.split()###按行读取文本文件，每行数据以列表形式返回
			data.append(eachline)
			line = f.readline()
		return data

#绘制折线图
def draw_ployline(x,y,color):
	plt.plot(x, y, linewidth=3, c=color)
	plt.tick_params(axis='both', labelsize=14)
	

if __name__ == '__main__':
	#filename = "/home/user/catkin_vloam/src/vloam/test/times_cost.txt"
	filename = "/home/user/catkin_vloam/src/vloam/test/times_cost_match.txt"

	data = read_text_data(filename)

	#plt.ylim(0, 13)
	plt.ylim(0, 60)
	#plt.title('Time cost of features detected', fontsize=14)
	plt.title('Time cost of features matched', fontsize=14)
	plt.xlabel('fram/(num)', fontsize=14)
	plt.ylabel('time/(ms)', fontsize=14)
	x=range(0, len(data[0]))
	draw_ployline(x,data[0],'red')
	draw_ployline(x,data[1],'green')
	#draw_ployline(x,data[2],'blue')
	plt.show()


