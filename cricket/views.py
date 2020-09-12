from django.shortcuts import render
from django.http import HttpResponse
import csv,io
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression, LogisticRegression
from .models import Destination
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def button(request):
    return render(request,'home.html')


def output(request):
    return render(request,'base.html')

def custom_accuracy(y_test,y_pred,thresold):
    right = 0

    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

def button1(request):
    obj=Destination()
    obj.teamA=request.GET["team1"]
    obj.teamB=request.GET["team2"]
    obj.city=request.GET["loc"]
    obj.tosswinning=request.GET["toss"]
    obj.tossdes=request.GET["tossds"]
    
    dataset = pd.read_csv('OutputOfAll.csv')
    X = dataset.iloc[:,[1,3,7]].values
    y = dataset.iloc[:, 7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lin = LinearRegression()
    lin.fit(X_train,y_train)
    
    y_pred = lin.predict(X_test)
    score = lin.score(X_test,y_test)
    obj.cus=custom_accuracy(y_test,y_pred,20)




    df = pd.read_csv('initialOutput.csv')
    TeamA = obj.teamA
    TeamB = obj.teamB

    Toss =  obj.tosswinning
    Toss_Decision = obj.tossdes

    Venue = obj.city
    
    playOffAandB=df[((df['TeamA']==obj.teamA)&(df['TeamB']==obj.teamB) | (df['TeamA']==obj.teamB)&(df['TeamB']==obj.teamA))]
    print(playOffAandB)

    playOffAandB = playOffAandB.sort_values(by = 'Date', ascending=[0])
    Awin=playOffAandB[(playOffAandB['Winner']==obj.teamA)]
    Awin1=playOffAandB[(playOffAandB['Winner']==obj.teamB)]
    if(len(Awin)>len(Awin1)):
        obj.winning=obj.teamA
    else:
        obj.winning=obj.teamB
    
    prevMatches = df[(df['Venue']==Venue)]

    prevMatches = prevMatches.sort_values(by = 'Date', ascending=[0])
    
    if Toss== obj.teamA:
        obj.winning=obj.teamA
        Awin11 = prevMatches[(prevMatches['Toss'] == prevMatches['Winner'])]
    else:
        obj.winning=obj.teamB
        Awin11 = prevMatches[(prevMatches['Toss'] != prevMatches['Winner'])]

    a=len(Awin)
    p=len(playOffAandB)
    
    r1=len(Awin1)

    a1=len(Awin11)
    p1=len(prevMatches)
    
    if p==0:
        if p1==0:
            obj.poss=0
            obj.poss1=0
            return render(request,'result.html',{'obj':obj})
    
    obj.poss=a/p
    obj.poss1=a1/p1
    obj.poss2=r1/p
    obj.rsq=score*obj.poss+1.5
    if(obj.poss>obj.poss2):
        obj.winning=obj.teamA
    else:
        obj.winning=obj.teamB
    return render(request,'result.html',{'obj':obj})


