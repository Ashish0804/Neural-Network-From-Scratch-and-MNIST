import neuralnet as nn
import numpy as np
import sys

def train(inputNodes=784,hiddenNodes=175,outputNodes=10,learningRate=0.2,epochs=7):
	n=nn.neuralNet(inputNodes,hiddenNodes,outputNodes,learningRate)
	trainingDataFile=open("mnist_train.csv",'r')
	trainingDataList=trainingDataFile.readlines()
	trainingDataFile.close()
	print("Training on (hn,lr,ep) {},{},{}".format(hiddenNodes,learningRate,epochs))
	for e in range(epochs):
	#	print("Traning epoch "+str(e))
		for record in trainingDataList:
			allValues=record.split(',')
			inputs = (np.asfarray(allValues[1:])/255.0*0.99)+0.01
			targets = np.zeros(outputNodes)+0.01
			targets[int(allValues[0])]=0.99
			n.train(inputs,targets)
			pass
		pass
	testDataFile=open("mnist_test.csv",'r')
	testDataList=testDataFile.readlines()
	testDataFile.close()

	scorecard=[]

	for record in testDataList:
		allValues=record.split(',')
		correctLabel=int(allValues[0])
		inputs = (np.asfarray(allValues[1:])/255.0*0.99)+0.01
		outputs=n.query(inputs)
		label=np.argmax(outputs)
		if(label==correctLabel):
			scorecard.append(1)
		else:
			scorecard.append(0)
			pass
		pass

	scorecardArray=np.asarray(scorecard)
	return scorecardArray.sum()/scorecardArray.size

def main(hiddenNodes=125,learningRate=0.25,epochs=4):
	inputNodes=784
	#hiddenNodes=int(sys.argv[2])
	outputNodes=10
	#learningRate=float(sys.argv[3])
	#epochs=int(sys.argv[4])
	logFile=open("log.csv",'w')
	logFile.write("Hidden Nodes,Learning Rate,Epochs,Perf\n")
	for hN in range(100,hiddenNodes,25):	
		for ep in range (3,epochs,2):
			for lr in np.arange(0.2,learningRate,0.05):
				#print("Training on (hn,lr,ep) {},{},{}".format(hN,lr,ep))
				perf=train(inputNodes,hN,outputNodes,lr,ep)
				line = "{},{},{},{}\n".format(hN,lr,ep,perf)
				logFile.write(line)
	logFile.close()

if __name__ == "__main__":
	#print("Usage:")
	#print("`python main.py` for hardcoded values")
	#print("`python main.py 1 HiddenNodes LearningRate Epochs` for testing optimum performance")
	#print("`python main.py 2 HiddenNodes LearningRate Epochs` for custom values")
	#print(sys.argv)
	if (len(sys.argv)==1):
		print("accuracy = "+ str(train()))
	elif (sys.argv[1]=="1"):
		main(int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]))
	elif (sys.argv[1]=="2"):
		print("accuracy = "+ str(train(784,int(sys.argv[2]),10,float(sys.argv[3]),int(sys.argv[4]))))
	else:
		print("Usage:")
		print("`python main.py` for hardcoded values")
		print("`python main.py 1 HiddenNodes LearningRate Epochs` for testing optimum performance")
		print("`python main.py 2 HiddenNodes LearningRate Epochs` for custom values")

