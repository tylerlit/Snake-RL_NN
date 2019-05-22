import csv
import numpy as np
import tflearn

class Net:
	def __init__(self):

		self.traindata, self.trainlabels = self.readdata()

		self.net = tflearn.input_data(shape=[None, 12])
		self.net = tflearn.fully_connected(self.net, 8, activation='sigmoid')
		self.net = tflearn.fully_connected(self.net, 8, activation='relu')
		self.net = tflearn.fully_connected(self.net, 4, activation='sigmoid')
		self.net = tflearn.regression(self.net)

		self.model = tflearn.DNN(self.net)
		self.model.fit(self.traindata, self.trainlabels, show_metric=True)

		# counter = 0
		# for x,y in zip(self.testdata, self.testlabels):
		# 	pred = self.model.predict([x])
		# 	if (y[0], y[1]) == (int(pred[0][0]), int(pred[0][1])):
		# 	 	counter += 1


	def getmodel(self):
		return self.model

	def readdata(self):
		data = []
		labels = []

		with open("data.csv", newline='') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				if row != []:
					add = []
					for i in row:
						add.append(int(i))
					data.append(add)
		with open("labels.csv", newline='') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				if row != []:
					add = []
					for i in row:
						add.append(int(i))
					labels.append(add)

		# testdata = data[:(len(data)//2)]
		# traindata = data[(len(data)//2):]
		# testlabels = labels[:(len(labels)//2)]
		# trainlabels = labels[(len(labels)//2):]

		# return 	np.asarray(traindata), np.asarray(trainlabels), \
		# np.asarray(testdata), np.asarray(testlabels)

		transform = []
		for x in labels:
			if (x[0], x[1]) == (0, 1):
				transform.append([1,0,0,0])
			elif (x[0], x[1]) == (0, -1):
				transform.append([0,1,0,0])
			elif (x[0], x[1]) == (-1, 0):
				transform.append([0,0,1,0])
			elif (x[0], x[1]) == (1, 0):
				transform.append([0,0,0,1])

		return data, transform

if __name__ == "__main__":
    print("initializing neural net")
    start = Net()