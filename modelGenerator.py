import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold


def Venue_Changes(teamA,teamB,venue):   #venue is changed to 1 for teamA, -1 for teamB and 0 for no team. 
	d = defaultdict(list)
	country=''
	with open ('stadium/stadiums', 'r') as f:
	    lines = f.read().splitlines()
	    length=len(lines)
	    i=0
	    while i<length:
	        line = lines[i]
	        if line == '$':
	        	country=lines[i+1]
	        	i+=1
	        else:
	        	d[country].append(line)
	        	i+=1

	if venue in (d[teamA]):
		return 1
	if venue in (d[teamB]):
		return -1
	return 0	


def Toss_Changes(teamA,teamB,Toss):
	if Toss == teamA:
		return 1
	return 0
		

def Toss_Decision_Changes(Toss,Toss_Decision):
	if ( ((Toss == 1) & (Toss_Decision == 'bat')) | ((Toss == 0) & (Toss_Decision == 'field')) ):
		return 1
	return 0	


def Win_Prob_Of_TeamA(df,teamA,teamB):

	playOffAandB=df[((df['TeamA']==teamA)&(df['TeamB']==teamB) | (df['TeamA']==teamB)&(df['TeamB']==teamA))]
	playOffAandB = playOffAandB.sort_values(by = 'Date', ascending=[0])
	# playOffAandB = playOffAandB.head(10)
	
	Awin=playOffAandB[(playOffAandB['Winner']==1)]
	a=len(Awin)
	p=len(playOffAandB)
	
	if p==0:
		return 0
	return a/p
	

def Win_prob_on_venue(df1,venue,Toss_Decision):

	prevMatches = df1[(df1['Venue']==venue)]
	prevMatches = prevMatches.sort_values(by = 'Date', ascending=[0])
	# prevMatches = prevMatches.head(10)
	
	#print prevMatches

	if Toss_Decision == 1:
		Awin = prevMatches[(prevMatches['Toss_Decision'] == prevMatches['Winner'])]
	else:
		Awin = prevMatches[(prevMatches['Toss_Decision'] != prevMatches['Winner'])]

	#print Awin
	a=len(Awin)
	p=len(prevMatches)
	
	if p==0:
		return 0
	return a/p	
	 
		
def strength_based_on_batBowl_avg(df,TeamA,TeamB):
	playOffAandB=df[((df['TeamA']==TeamA)&(df['TeamB']==TeamB) | (df['TeamA']==TeamB)&(df['TeamB']==TeamA))]
	playOffAandB = playOffAandB.sort_values(by = 'Date', ascending=[0])
	playOffAandB = playOffAandB["Strength"].iloc[0]
	
	return playOffAandB


def pastPerformance(df1,teamA,teamB,bat_avg):
	prevA = df1[((df1['TeamA']==teamA) | (df1['TeamB']==teamA))]
	prevA = prevA.sort_values(by = 'Date', ascending=[0])
	prevA = prevA.head(10)
	
	form_A = 0
	cntA = 0
	for index, row in prevA.iterrows():
		name = str(row['MatchID'])+'.csv'#"657643" #657645
		df=pd.read_csv("Dataset/PlayerInfo/"+name)
		df['Bat_Avg'] = df['Bat_Avg'].replace('-', bat_avg)
		total_A = 0
		cntA = cntA + 1
		team_list=df[(df['Country']==teamA)]
		for index1, row1 in team_list.iterrows():
			total_A=total_A+float(row1['Bat_Avg'])
		form_A = form_A + total_A/11

	prevB = df1[((df1['TeamA']==teamB) | (df1['TeamB']==teamB))]
	prevB = prevB.sort_values(by = 'Date', ascending=[0])
	prevB = prevB.head(10)

	form_B = 0
	cntB = 0
	for index, row in prevB.iterrows():
		name = str(row['MatchID'])+'.csv'#"657643" #657645
		df=pd.read_csv("Dataset/PlayerInfo/"+name)
		df['Bat_Avg'] = df['Bat_Avg'].replace('-', bat_avg)
		total_B = 0
		cntB = cntB + 1
		team_list=df[(df['Country']==teamB)]
		for index1, row1 in team_list.iterrows():
			total_B=total_B+float(row1['Bat_Avg'])
		form_B = form_B + total_B/11

	# print form_A, form_B, df1.loc[i, 'MatchID'] 
	if cntA == 0:
		cntA = 1
		formA = bat_avg
	if cntB == 0:
		cntB = 1
		formB = bat_avg

	return form_A/cntA - form_B/cntB	



def testPredicit(df1,testData,TeamA,TeamB):
	df1 = df1[((df1['TeamA']==TeamA)&(df1['TeamB']==TeamB) | (df1['TeamA']==TeamB)&(df1['TeamB']==TeamA))]
	predictors = ['Toss', 'Toss_Decision','Venue', 'HTH', 'WinningPerDes','Strength','latest_form']
	alg = LogisticRegression(random_state=1)
	df = df1[['Toss', 'Toss_Decision', 'Venue', 'HTH', 'WinningPerDes', 'Strength','latest_form','Winner']]
	train_predictors = (df[predictors])
	train_target = df["Winner"]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(testData)
	return test_predictions[0] 


# main Function

def startPrediction(teamA_input,teamB_input,venue_input,toss_input,tossDecision_input):

	df = pd.read_csv('OutputOfAll.csv')

	if teamB_input < teamA_input:
		teamB_input,teamA_input = teamA_input,teamB_input

	TeamA = teamA_input
	TeamB = teamB_input
	Toss = toss_input
	Toss_Decision = tossDecision_input
	Venue = venue_input


	Venue = Venue_Changes(TeamA,TeamB,Venue)
	Toss = Toss_Changes(TeamA,TeamB,Toss)
	Toss_Decision = Toss_Decision_Changes(Toss,Toss_Decision)

	HTH = Win_Prob_Of_TeamA(df,TeamA,TeamB)

	WinningPerDes = Win_prob_on_venue(df,Venue,Toss_Decision)

	bat_avg = 22.6046511628
	bowl_avg = 29.7670682731

	Strength = strength_based_on_batBowl_avg(df,TeamA,TeamB)

	latest_form = pastPerformance(df,TeamA,TeamB,bat_avg)

	print ("teamA : " +  TeamA)
	print (" ")
	print ("teamB :"  + TeamB)
	print(" ")
	print ( "winning probability of TeamA based on previous matches : " + str(HTH))
	print (" ")
	print ( "winning probability of Team batting first : " + str(WinningPerDes))
	print(" ")
	print ("Team A Performance - Team B Performance : " + str(latest_form))
	print(" ")
	print ("Performance is calculated based on team's batting average")
	print(" ")
	dict = {'Toss':Toss, 'Toss_Decision':Toss_Decision, 'Venue':Venue, 'HTH':HTH, 'WinningPerDes':WinningPerDes, 'Strength':Strength , 'latest_form':latest_form}

	testData = pd.DataFrame(dict,index = ["result"])

	if testPredicit(df,testData,TeamA,TeamB) == 1:
		return teamA_input
	return teamB_input	

