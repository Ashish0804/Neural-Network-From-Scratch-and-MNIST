import numpy as np
import scipy.special

class neuralNet:

	def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
		self.inputNodes=inputNodes
		self.hiddenNodes=hiddenNodes
		self.outputNodes=outputNodes
		self.learningRate=learningRate
		self.weightsInputHidden=np.random.normal(0.0,pow(self.hiddenNodes,-0.5),(self.hiddenNodes,self.inputNodes))
		self.weightsHiddenOutput=np.random.normal(0.0,pow(self.outputNodes,-0.5),(self.outputNodes,self.hiddenNodes))
		self.activationFunction=lambda x:scipy.special.expit(x)
		pass

	def train(self,inputList,targetList):
		inputs=np.array(inputList,ndmin=2).T
		targets=np.array(targetList,ndmin=2).T
		hiddenInputs=np.dot(self.weightsInputHidden,inputs)
		hiddenOutputs=self.activationFunction(hiddenInputs)
		finalInputs=np.dot(self.weightsHiddenOutput,hiddenOutputs)
		finalOutputs=self.activationFunction(finalInputs)
		outputErrors=targets-finalOutputs
		hiddenErrors=np.dot(self.weightsHiddenOutput.T,outputErrors)
		self.weightsHiddenOutput += self.learningRate * np.dot((outputErrors*finalOutputs*(1.0-finalOutputs)),np.transpose(hiddenOutputs))
		self.weightsInputHidden += self.learningRate * np.dot((hiddenErrors*hiddenOutputs*(1.0-hiddenOutputs)),np.transpose(inputs))
		pass

	def query(self,inputList):
		inputs=np.array(inputList,ndmin=2).T
		hiddenInputs=np.dot(self.weightsInputHidden,inputs)
		hiddenOutputs=self.activationFunction(hiddenInputs)
		finalInputs=np.dot(self.weightsHiddenOutput,hiddenOutputs)
		finalOutputs=self.activationFunction(finalInputs)
		return finalOutputs
		

