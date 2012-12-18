#Author: Krishna Sai P B V
#Date: 10/08/2012

#Input to this file is nothing
#Output is week learner predictor with boosting applied
#until obtaining convergence (displays error rate)
#For the convergent value, calculates AUC (Area under the ROC curve)

#!/usr/bin python
import string,re,sys,math, operator, random
from operator import itemgetter
from collections import defaultdict
from random import randint

spamDataSet=open('spambase.data','r')
spamData=spamDataSet.readlines()

#number of folds.
k=10

#Array storing the input
kFold = [[]*k for x in xrange(k)]

#calculating kFold by parsing through
#the given spam data file.
i=0
for line in spamData:
	j=i%10;
	line = line.strip('\r\n')
	line = line.split(',')
	kFold[j].append(line)
	i=i+1

#Training and testing set arrays
trainingSet=[]
testingSet=[]

#Go through the input and get training and testing
#data
for i in range(0,len(kFold)):
	if(i!=1):
		for k in range(0,len(kFold[i])):
			flagArray=[]
			for feature in kFold[i][k]:
				toAppend=float(feature)
				flagArray.append(toAppend)
			trainingSet.append(flagArray)
	else:
		for k in range(0,len(kFold[i])):
			flagArray=[]
			for feature in kFold[i][k]:
				toAppend=float(feature)
				flagArray.append(toAppend)
			testingSet.append(flagArray)

#Array storing sorted features
featureSorted=[]

#Sort the features and remove duplicates
def featuresSort():
	for feature in range(0,len(trainingSet[0])):
		featureList=[]
		for i in range(0,len(trainingSet)):
			featureList.append(float(trainingSet[i][feature]))
		featureList=set(featureList)
		featureList=sorted(featureList)
		featureSorted.append(featureList)


#Call the featuresSort function to
#remove duplicates and sort the feature values		
featuresSort()

#store threshold
featureThreshold=[[] * len(featureSorted) for x in xrange(len(featureSorted))]

#calculate thresholds
def calThreshold():
	for i in range(0,len(featureSorted)):
		for k in range(0,len(featureSorted[i])-1):
			if(k==0):
				firstValue= float((featureSorted[i][k]) - (0.001))
				featureThreshold[i].append(firstValue)	
				featureThreshold[i].append(float(featureSorted[i][k] + featureSorted[i][k+1]) / 2.0)
			else:
				featureThreshold[i].append(float(featureSorted[i][k] + featureSorted[i][k+1]) / 2.0)				
		#Add two thresholds one is less than all, the other is greater
		#than all the current thresholds
		lastValue=featureThreshold[i][len(featureThreshold[i])-1]
		featureThreshold[i].append(lastValue+0.1)

#Call the threshold calculating function	
calThreshold()

#Append the weight to each email
for email in trainingSet:
	email.append(1.0/ 4141.0)

#
featureToTrainingEmails={}

#Sort the input based on each feature
def sortTrainingData():
	for i in range(0,len(featureSorted)-1):
		featureToTrainingEmails[i] = sorted(trainingSet,key=itemgetter(i))

sortTrainingData()

#return sign of the function
def sign(i):
	if(i<0):
		return 0;
	else:
		return 1;

#Calculate training error
def calTrainErrorHelper(trainingError):
	emailCount=0
	trainingErrorValue=0
	for email in trainingSet:
		if(sign(trainingError[emailCount]) != email[57]):
			trainingErrorValue= trainingErrorValue + 1
		emailCount+=1
	#print trainingErrorValue
	return float(trainingErrorValue) / len(trainingSet)

#Calculate testing error
def calTestErrorHelper(testErrorSet):
	emailCount=0
	testErrorValue=0
	for email in testingSet:
		if(sign(testErrorSet[emailCount]) != email[57]):
			testErrorValue= testErrorValue + 1
		emailCount+=1
	return float(testErrorValue) / len(testingSet)

#update distributions
def updateDistributions(minFeature,minThreshold,minErrorRate,confidenceHere,trainingError):
	for featureRange in range(0,1):
		emailCount=0
		totalValue=0.0
		for email in featureToTrainingEmails[featureRange]:
			previousWeight=email[58]
			if(email[minFeature] > minThreshold):
				if(email[57] == 1):
					distHere = math.sqrt(float(minErrorRate) / float(1-minErrorRate))	
					flag=previousWeight * distHere
					totalValue+= flag
					email[58]=flag
				else:
					distHere = math.sqrt(float(1-minErrorRate)/float(minErrorRate))
					flag=previousWeight * distHere
					totalValue+= flag
					email[58]=flag
			else:
				if(email[57] == 1):
					distHere = math.sqrt(float(1-minErrorRate)/float(minErrorRate))
					flag=previousWeight * distHere
					totalValue+= flag
					email[58]=flag
				else:
					distHere = math.sqrt(minErrorRate / float(1-minErrorRate))
					flag=previousWeight * distHere
					totalValue+= flag
					email[58]=flag
			emailCount+=1

		for i in range(0,len(featureToTrainingEmails[featureRange])):
			featureToTrainingEmails[featureRange][i][58] = float(featureToTrainingEmails[featureRange][i][58]) / totalValue

#cal training error
def calTrainingError(minFeature, minThreshold,confidenceHere,trainingError):
	emailCount=0
	for email in trainingSet:
		if(email[minFeature] > minThreshold):
			trainingError[emailCount]+= confidenceHere * 1.0
		else:
			trainingError[emailCount]+= confidenceHere * -1.0
		emailCount+=1
	trainErrorValue=calTrainErrorHelper(trainingError)
	return trainErrorValue, trainingError

#cal testing error
def calTestError(minFeature, minThreshold,confidenceHere,testErrorSet):
	emailCount=0
	for email in testingSet:
		if(email[minFeature] > minThreshold):
			testErrorSet[emailCount] += confidenceHere * 1.0
		else:
			testErrorSet[emailCount] += confidenceHere * -1.0
		emailCount+=1
	testErrorValue = calTestErrorHelper(testErrorSet);
	return testErrorValue, testErrorSet

#calculating AUC
def calAUC(fpr, tpr):
	sumValue=0
	for i in range(1,len(fpr)):
		sumValue+= ((fpr[i]-fpr[i-1]) * (tpr[i] + tpr[i-1]))
	#print "AUC: ",(0.5*sumValue)
	return (0.5*sumValue)

#calculate ROC
def calROC(scores,unsortedValues,printFlag):
	i=0
	dataSpam=[]
	nspam=0
	spam=0
	for email in testingSet:
		if(int(email[57])==0):
			dataSpam.append(0)
			nspam+=1
		else:
			dataSpam.append(1)
			spam+=1
		i+=1
	j=0
	fpr=[]
	tpr=[]
	fnr=[]
	tnr=[]
	for i in range(0,len(scores)):
		fpr.append(0)
		tpr.append(0)
		fnr.append(0)
		tnr.append(0)
	for i in range(0,len(scores)):
		fp=0
		tp=0
		tn=0
		fn=0
		for k in range(0,len(scores)):
			if(i!=k):
				if(unsortedValues[k]>=scores[i]):
					if(dataSpam[k]==0):
						fp=fp+1
					else:
						tp=tp+1
				else:
					if(dataSpam[k]==1):
						fn=fn+1
					else:
						tn=tn+1
		fpr[i]=fp
		tpr[i]=tp
		tnr[i]=tn
		fnr[i]=fn
	for i in range(0,len(fpr)):
		fpr[i]=float(fpr[i])/float(fpr[i] + tnr[i])
		tpr[i]=float(tpr[i])/float(tpr[i] + fnr[i])
		
	if(printFlag):
		for i in range(0,len(fpr)):
			print fpr[i], tpr[i]
	return calAUC(fpr,tpr)

#check convergence
def convergence(aucTotal):
	if(len(aucTotal) >= 2):
		if((aucTotal[len(aucTotal)-2] - aucTotal[len(aucTotal)-1]) > 0.00001 and aucTotal[len(aucTotal)-1] >= 0.97):
			return True;
		else:
			return False;
	else:
		return False;

#Calculate error rates for a optimal decision stump
def calErrorRateForOptimal():
	trainingError=[0.0] * len(trainingSet)
	testingErrorSet=[0.0] * len(testingSet)
	aucTotal=[]
	#for boostRound in range(0,100):
	boostRound=0
	print "Calculating for optimal data stumps:"
	while(not convergence(aucTotal)):
		frNo=0
		errorRatesAllFeatures=[]
		minErrorRateToCompare= -123456789
		minErrorRate=0
		minThreshold=0
		minFeature=0
		for featureRange in range(0,len(featureSorted)-1):
			thresholds=featureThreshold[featureRange]
			errorRate=0
			emailCount=0
			for i in range(0,len(thresholds)):
				if(i==0):
					counter=0
					for email in featureToTrainingEmails[featureRange]:
						if(email[featureRange] > thresholds[i]):
							flagLength=58	
							weightOfDataPoint= email[flagLength]
							if(email[57]==0):
								errorRate+=weightOfDataPoint
						counter+=1
				else:
					if(emailCount < len(trainingSet)-1):
						if(featureToTrainingEmails[featureRange][emailCount][featureRange] == featureToTrainingEmails[featureRange][emailCount+1][featureRange]):	
							flag1=featureToTrainingEmails[featureRange][emailCount][featureRange] 
							flag2=featureToTrainingEmails[featureRange][emailCount+1][featureRange] 		
							while(flag1 == flag2):					
								email=featureToTrainingEmails[featureRange][emailCount]
								flagLength=58
								weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
								if(email[57]==0.0):
									errorRate-=weightOfDataPoint
								else:
									errorRate+=weightOfDataPoint
								emailCount+=1
								flag1=featureToTrainingEmails[featureRange][emailCount-1][featureRange]
								if(emailCount<len(trainingSet)):
									flag2=featureToTrainingEmails[featureRange][emailCount][featureRange]
								else:
									flag2=sys.maxint;
						else:				
							email=featureToTrainingEmails[featureRange][emailCount]
							flagLength=58
							weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
							if(email[57]==0):
								errorRate-=weightOfDataPoint
							else:
								errorRate+=weightOfDataPoint
							emailCount+=1
					else:
						email=featureToTrainingEmails[featureRange][emailCount]
						flagLength=58
						weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
						if(email[57]==0):
							errorRate-=weightOfDataPoint
						else:
							errorRate+=weightOfDataPoint
						emailCount+=1
				errorRateToCompare=math.fabs(0.5-errorRate);
				if(errorRateToCompare > minErrorRateToCompare):
					minErrorRateToCompare=errorRateToCompare
					minFeature=featureRange
					minErrorRate = errorRate
					minThreshold = thresholds[i]
				#print thresholds[i], errorRate, minFeature,minThreshold,minErrorRate
		confidenceHere = 0.5 * math.log(float(1 - minErrorRate) / float(minErrorRate))		
		updateDistributions(minFeature,minThreshold,minErrorRate,confidenceHere,trainingError)
		trainingErrorValue,trainingError=calTrainingError(minFeature, minThreshold,confidenceHere,trainingError)
		testError,testingErrorSet=calTestError(minFeature,minThreshold,confidenceHere,testingErrorSet)
		testingErrorSet_sorted=sorted(testingErrorSet,reverse=True)
		aucHere=calROC(testingErrorSet_sorted,testingErrorSet,0)
		aucTotal.append(aucHere)
		#print boostRound, aucHere
		print boostRound,minFeature,minThreshold, minErrorRate, trainingErrorValue, testError, aucHere
		#print boostRound,minErrorRate
		#print boostRound, trainingErrorValue
		#print boostRound, testError
		boostRound+=1
	calROC(testingErrorSet_sorted,testingErrorSet,1)

#cal function to calculate error rate
calErrorRateForOptimal()

#Calculate error rates for a optimal decision stump
def calErrorRateForRandom():
	trainingError=[0.0] * len(trainingSet)
	testingErrorSet=[0.0] * len(testingSet)
	boostRound=0
	aucTotal=[]
	print "Calculating for random data stumps:"
	while(not convergence(aucTotal)):
	#for boostRound in range(0,150):
		frNo=0
		errorRatesAllFeatures=[]
		minErrorRate=sys.maxint
		minThreshold=0
		minFeature=0
		for featureRange in range(0,len(featureSorted)-1):
			thresholds=featureThreshold[featureRange]
			errorRate=0
			emailCount=0
			for i in range(0,len(thresholds)):
				if(i==0):
					counter=0
					for email in featureToTrainingEmails[featureRange]:
						if(email[featureRange] > thresholds[i]):
							flagLength=58	
							weightOfDataPoint= email[flagLength]
							if(email[57]==0):
								errorRate+=weightOfDataPoint
						counter+=1
				else:
					if(emailCount < len(trainingSet)-1):
						if(featureToTrainingEmails[featureRange][emailCount][featureRange] == featureToTrainingEmails[featureRange][emailCount+1][featureRange]):	
							flag1=featureToTrainingEmails[featureRange][emailCount][featureRange] 
							flag2=featureToTrainingEmails[featureRange][emailCount+1][featureRange] 		
							while(flag1 == flag2):					
								email=featureToTrainingEmails[featureRange][emailCount]
								flagLength=58
								weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
								if(email[57]==0.0):
									errorRate-=weightOfDataPoint
								else:
									errorRate+=weightOfDataPoint
								emailCount+=1
								flag1=featureToTrainingEmails[featureRange][emailCount-1][featureRange]
								if(emailCount<len(trainingSet)):
									flag2=featureToTrainingEmails[featureRange][emailCount][featureRange]
								else:
									flag2=sys.maxint;
						else:				
							email=featureToTrainingEmails[featureRange][emailCount]
							flagLength=58
							weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
							#if(len(email) > 1):
							if(email[57]==0):
								errorRate-=weightOfDataPoint
							else:
								errorRate+=weightOfDataPoint
							emailCount+=1
					else:
						email=featureToTrainingEmails[featureRange][emailCount]
						flagLength=58
						weightOfDataPoint= featureToTrainingEmails[featureRange][emailCount][flagLength]
						if(email[57]==0):
							errorRate-=weightOfDataPoint
						else:
							errorRate+=weightOfDataPoint
						emailCount+=1
					flagArray=[]
					flagArray.append(featureRange)
					flagArray.append(thresholds[i])
					flagArray.append(errorRate)
					errorRatesAllFeatures.append(flagArray)
		choosenValue=randint(0,len(errorRatesAllFeatures))
		minFeature = errorRatesAllFeatures[choosenValue][0]
		minThreshold=errorRatesAllFeatures[choosenValue][1]
		minErrorRate=errorRatesAllFeatures[choosenValue][2]
		confidenceHere = 0.5 * math.log(float(1 - minErrorRate) / float(minErrorRate))		
		updateDistributions(minFeature,minThreshold,minErrorRate,confidenceHere,trainingError)
		trainingErrorValue,trainingError=calTrainingError(minFeature, minThreshold,confidenceHere,trainingError)
		testError,testingErrorSet=calTestError(minFeature,minThreshold,confidenceHere,testingErrorSet)
		testingErrorSet_sorted=sorted(testingErrorSet,reverse=True)
		aucHere=calROC(testingErrorSet_sorted,testingErrorSet,0)
		aucTotal.append(aucHere)
		print boostRound,minFeature,minThreshold, minErrorRate, trainingErrorValue, testError, aucHere
		#print boostRound, minErrorRate
		boostRound+=1
	calROC(testingErrorSet_sorted,testingErrorSet,1)


#cal function to calculate error rate
calErrorRateForRandom()





