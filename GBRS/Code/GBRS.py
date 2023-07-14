import copy
import csv
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


class GranularBall:
	"""class of the granular ball"""

	def __init__(self, data, attribute):
		"""
		:param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
		and each of the preceding columns corresponds to a feature
		"""
		self.data = data[:, :]
		self.attribute = attribute
		self.data_no_label = data[:, attribute]
		self.num, self.dim = self.data_no_label.shape
		self.center = self.data_no_label.mean(0)
		self.label, self.purity, self.r = self.__get_label_and_purity_and_r()

	def __get_label_and_purity_and_r(self):
		"""
		:return: the label and purity of the granular ball.
		"""
		count = Counter(self.data[:, -2])

		label = max(count, key=count.get)
		purity = count[label] / self.num
		arr = np.array(self.data_no_label) - self.center
		ar = np.square(arr)
		a = np.sqrt(np.sum(ar, 1))
		r = max(a)  # ball's radius is max disitient
		return label, purity, r

	def split_2balls(self):
		"""
	    split into two ball
		"""
		# label_cluster = KMeans(X=self.data_no_label, n_clusters=2)[1]
		# print(self.data_no_label.shape,self.num,self.dim)#
		labs = set(self.data[:, -2].tolist())
		# print("labs",labs)
		i1 = 9999
		ti1 = -1
		ti0 = -1
		Balls = []
		i0 = 9999
		dol = np.sum(self.data_no_label, axis=1)
		if len(labs) > 1:
			for i in range(len(self.data)):
				if self.data[i, -2] == 1 and dol[i] < i1:
					i1 = dol[i]
					ti1 = i
				elif self.data[i, -2] != 1 and dol[i] < i0:
					i0 = dol[i]
					ti0 = i
			ini = self.data_no_label[[ti0, ti1], :]
			clu = KMeans(n_clusters=2, init=ini).fit(self.data_no_label)  # select primary sample center
			label_cluster = clu.labels_
			if len(set(label_cluster)) > 1:
				ball1 = GranularBall(self.data[label_cluster == 0, :], self.attribute)
				ball2 = GranularBall(self.data[label_cluster == 1, :], self.attribute)
				Balls.append(ball1)
				Balls.append(ball2)
			else:
				Balls.append(self)
		else:

			Balls.append(self)
		return Balls


def funtion(ball_list, minsam):
	Ball_list = ball_list
	Ball_list = sorted(Ball_list, key=lambda x: -x.r, reverse=True)
	ballsNum = len(Ball_list)
	j = 0
	ball = []
	while True:
		if len(ball) == 0:
			ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])
			j += 1
		else:
			flag = False
			for index, values in enumerate(ball):
				if values[2] != Ball_list[j].label and (
						np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
						values[1] + Ball_list[j].r) and Ball_list[j].r > 0 and Ball_list[j].num >= minsam / 2 and \
						values[3] >= minsam / 2:
					balls = Ball_list[j].split_2balls()
					if len(balls) > 1:
						Ball_list[j] = balls[0]
						Ball_list.append(balls[1])
						ballsNum += 1
					else:
						Ball_list[j] = balls[0]
			if flag == False:
				# print(8)
				ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])
				j += 1
		if j >= ballsNum:
			break
	ball = []
	j = 0
	#if two ball's label is different and overlapped，in this step,we can continut split positive domain can keep boundary region don't change, but this measure is unnecessary
	# while 1:
	# 	if len(ball) == 0:
	# 		ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num,Ball_list[j]])
	# 		j += 1
	# 	else:
	# 		flag = False
	# 		for index, values in enumerate(ball):
	# 			if values[2] != Ball_list[j].label and (
	# 					np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
	# 					values[1] + Ball_list[j].r) and Ball_list[j].r > 0 and (Ball_list[j].purity>0.995 or values[-1].purity>0.995):
	# 				if(values[-1].purity<0.99):
	# 					balls = Ball_list[j].split_2balls()
	# 					if len(balls) > 1:
	# 						Ball_list[j] = balls[0]
	# 						Ball_list.append(balls[1])
	# 						ballsNum += 1
	# 					else:
	# 						Ball_list[j] = balls[0]
	# 				elif Ball_list[j].purity<0.99:
	# 					balls = values[-1].split_2balls()
	# 					if len(balls) > 1:
	# 						values[-1] = balls[0]
	# 						Ball_list.append(balls[1])
	# 						ballsNum += 1
	# 					else:
	# 						Ball_list[j] = balls[0]
	# 		if flag == False:
	# 			# print(8)
	# 			ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num,Ball_list[j]])
	# 			j += 1
	# 	if j >= ballsNum:
	# 		break
	return Ball_list


def overlap_resolve(ball_list, data, attributes_reduction, min_sam):
	Ball_list = funtion(ball_list, min_sam)  # conitinue to split ball which are overlapped
	# do last overlap for granular ball aimed raise ball's quality
	while True:
		init_center = []  # ball's center
		Ball_num1 = len(Ball_list)
		for i in range(len(Ball_list)):
			init_center.append(Ball_list[i].center)
		ClusterLists = KMeans(init=np.array(init_center), n_clusters=len(Ball_list)).fit(data[:, attributes_reduction])
		data_label = ClusterLists.labels_
		ball_list = []
		for i in set(data_label):
			ball_list.append(GranularBall(data[data_label == i, :], attributes_reduction))
		Ball_list = funtion(ball_list, min_sam)
		Ball_num2 = len(Ball_list)  # get ball numbers
		if Ball_num1 == Ball_num2:  # stop until ball's numbers don't change
			break
	return Ball_list


class GBList:
	"""class of the list of granular ball"""

	def __init__(self, data=None, attribu=[]):
		self.data = data[:, :]
		self.attribu = attribu
		self.granular_balls = [GranularBall(self.data, self.attribu)]  # gbs is initialized with all data

	def init_granular_balls(self, purity=0.996, min_sample=1):
		"""
		Split the balls, initialize the balls list.
		purty=1,min_sample=2d
		:param purity: If the purity of a ball is greater than this value, stop splitting.
		:param min_sample: If the number of samples of a ball is less than this value, stop splitting.
		"""
		ll = len(self.granular_balls)
		i = 0
		while True:
			if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
				split_balls = self.granular_balls[i].split_2balls()
				if len(split_balls) > 1:
					self.granular_balls[i] = split_balls[0]
					self.granular_balls.append(split_balls[1])
					ll += 1
				else:
					i += 1
			else:
				i += 1
			if i >= ll:
				break
		ball_lists = self.granular_balls
		Bal_List = overlap_resolve(ball_lists, self.data, self.attribu, min_sample)  # do overlap
		self.granular_balls = Bal_List
		self.get_data()
		self.data = self.get_data()

	def get_data_size(self):
		return list(map(lambda x: len(x.data), self.granular_balls))

	def get_purity(self):
		return list(map(lambda x: x.purity, self.granular_balls))

	def get_center(self):
		"""
		:return: the center of each ball.
		"""
		return np.array(list(map(lambda x: x.center, self.granular_balls)))

	def get_r(self):
		"""
		:return: return radius r
		"""
		return np.array(list(map(lambda x: x.r, self.granular_balls)))

	def get_data(self):
		"""
		:return: Data from all existing granular balls in the GBlist.
		"""
		list_data = [ball.data for ball in self.granular_balls]
		return np.vstack(list_data)

	def del_ball(self, purty=0., num_data=0):
		# delete ball
		T_ball = []
		for ball in self.granular_balls:
			if ball.purity >= purty and ball.num >= num_data:
				T_ball.append(ball)
		self.granular_balls = T_ball.copy()
		self.data = self.get_data()

	def R_get_center(self, i):
		# get ball's center
		attribu = self.attribu
		attribu.append(i)
		centers = []
		for ball in range(self.granular_balls):
			center = []
			data_no_label = ball.data[:, attribu]
			center = data_no_label.mean(0)
			centers.append(center)
		return centers


def attribute_reduce(data, pur=1, d2=2):
	"""
	:param data: dataset
	:param pur: purity threshold
	:param d2: min_samples, the default value is 2
	:return: reduction attribute
	"""
	bal_num = -9999
	attribu = []
	re_attribu = [i for i in range(len(data[0]) - 2)]
	while len(re_attribu):
		N_bal_num = -9999
		N_i = -1
		N_attribu = copy.deepcopy(attribu)
		for i in re_attribu:
			N_attribu = copy.deepcopy(attribu)
			N_attribu.append(i)
			gb = GBList(data, N_attribu)  # create the list of granular balls
			gb.init_granular_balls(purity=pur, min_sample=2 * (len(data[0]) - d2))  # initialize the list
			ball_list1 = gb.granular_balls
			# for bal in ball_list1:
			# 	if bal.purity<=0.999:
			# 		print(bal.center,bal.r)
			# 	else:
			# 		print(bal.center,bal.r,bal.label)
			Pos_num = 0
			for ball in ball_list1:
				if ball.purity >= 1:
					Pos_num += ball.num  # find the current  domain samples
			if Pos_num > N_bal_num:
				N_bal_num = Pos_num
				N_i = i
		if N_bal_num >= bal_num:
			bal_num = N_bal_num
			attribu.append(N_i)
			re_attribu.remove(N_i)
		else:
			return attribu
	return attribu


def mean_std(a):
	# calucate average and standard
	a = np.array(a)
	std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
	return a.mean(), std


if __name__ == "__main__":
	datan = ["wine"]
	for name in datan:
		with open(r"E:\lianxiaoyu\xia\GBRS2\GBRS\Result\\" + name + ".csv", "w", newline='', encoding="utf-8") as jg:
			writ = csv.writer(jg)
			df = pd.read_csv(r"E:\lianxiaoyu\xia\GBRS2\GBRS\DataSet\\" + name + ".csv")
			data = df.values
			numberSample, numberAttribute = data.shape
			minMax = MinMaxScaler()
			U = np.hstack((minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))
			C = list(np.arange(0, numberAttribute - 1))
			D = list(set(U[:, -1]))
			index = np.array(range(0, numberSample)).reshape(numberSample, 1)
			sort_U = np.argsort(U[:, 0:-1], axis=0)
			U1 = np.hstack((U, index))
			index = np.array(range(numberSample)).reshape(numberSample, 1)  # column of index
			data_U = np.hstack((U, index))  # Add the index column to the last column of the data
			purty = 1
			clf = KNeighborsClassifier(n_neighbors=5)
			orderAttributes = U[:, -1]
			mat_data = U[:, :-1]
			maxavg = -1
			maxStd = 0
			maxRow = []
			for i in range((int)(numberAttribute)):
				nums = i
				Row = attribute_reduce(data_U, pur=purty, d2=nums)
				writ.writerow(["FGBNRS", Row])
				print("Row:", Row)
				mat_data = U[:, Row]
				scores = cross_val_score(clf, mat_data, orderAttributes, cv=5)
				avg, std = mean_std(scores)
				if maxavg < avg:
					maxavg = avg
					maxStd = std
					maxRow = copy.deepcopy(Row)
			print("pre", maxavg)
			print("row", maxRow)
