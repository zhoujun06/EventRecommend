#!/usr/bin/env python
from __future__ import division
from dateutil.parser import parse
from copy import copy
import os
import csv 
import random
import numpy  as np
import pandas as pd
from sklearn import svm
from sklearn import tree 
from sklearn.metrics import classification_report

#class GenFeature:
#	def __init__(self):
#		self._user_feature_dict = {}
#		self._event_feature_dict = {}
#		self._user_event_feature_dict = {}
#
#	def gen_user_feature(self):
#		
#

def isNotNull(string):
	return string is not None and len(str(string)) > 0

def main():
	data_path = '../'
	train_file = data_path + 'train.csv'
	event_file = data_path + 'events_valid.csv'
	user_file = data_path + 'users.csv'
	attend_file = data_path + 'event_attendees_stat.csv'
	user_like_cnt_file = data_path + 'user_like_cnt_p'
	user_friends_file = data_path + 'user_friends.csv'
	event_attend_file = data_path + 'event_attendees.csv'

	train = pd.read_csv(train_file, converters={"timestamp":parse})
	events = pd.read_csv(event_file, converters={'start_time':parse})
	users = pd.read_csv(user_file)
	pop = pd.read_csv(attend_file)
	#user_like_cnt = pd.read_csv(user_like_cnt_file)
	user_friends = pd.read_csv(user_friends_file)
	event_attend = pd.read_csv(event_attend_file)

	#userinfo and user like cnt
	#cnt_user = pd.merge(users, user_like_cnt, on='user_id')
	#training user info
	train_user = pd.merge(train, users, left_on='user', right_on='user_id')
	tue = pd.merge(train_user, events, left_on='event', right_on='event_id')
	tue_pop = pd.merge(tue, pop, left_on='event', right_on='event')

	#time_diff, in unit of hours
	tue_pop['time_diff']=0
	#location of user and event, using simple string contain match
	tue_pop['loc_match']=0
	tue_pop['class'] = tue_pop['interested']
	tue_pop['start_hour']=0
	tue_pop['friends_yes']=0

	#building dicts of user_friends and event_attends
	#in order for fast proceccing in the following for loop
	user_friend_dict = {}
	event_attend_dict = {}
	ufd_fname = 'user_friend_dict.txt'
	ead_fname = 'event_attend_dict.txt'

	#if os.path.isfile(ufd_fname):
	#	user_friend_dict = eval(open(ufd_fname,'r').read())
	#	event_attend_dict = eval(open(ead_fname,'r').read())
	#else:
	for row_index, row in user_friends.iterrows():
		uid = int(row['user'])
		user_friend_dict[uid]=[]
		if isNotNull(row['friends']):
			frds = str(row['friends']).split()
			user_friend_dict[uid]=frds

	for row_index, row in event_attend.iterrows():
		eid = int(row['event'])
		event_attend_dict[eid]=[]
		if isNotNull(row['yes']):
			atds = str(row['yes']).split()
			event_attend_dict[eid]=atds

	#ufd_file = open(ufd_fname,'w')
	#ufd_file.write(str(user_friend_dict))
	#ead_file = open(ead_fname,'w')
	#ead_file.write(str(event_attend_dict))

	for row_index, row in tue_pop.iterrows():
		#list of friends of the user
		#friends = user_friends.select(lambda i: user_friends.irow(i)['user']==row['user'])['friends'].str.split().values[0]
		#list of attendees of the event
		#attends = event_attend.select(lambda i: event_attend.irow(i)['event']==row['event'])['yes'].str.split().values[0]
		#number of friends who attend this event
		uid = int(row['user'])
		eid = int(row['event'])
		tue_pop['friends_yes'][row_index]=len([w for w in user_friend_dict[uid] if w in event_attend_dict[eid]])

		if row['birthyear'].isdigit():
			tue_pop['birthyear'][row_index] = 2012-int(row['birthyear'])
		else:
			tue_pop['birthyear'][row_index] = 0 

		if row['gender'] == 'male':
			tue_pop['gender'][row_index] = 1
		else:
			tue_pop['gender'][row_index] = 0

		tue_pop['start_hour'][row_index] = row['start_time'].hour 
		diff = row['start_time']-row['timestamp']
		tue_pop['time_diff'][row_index]=int(diff.total_seconds()/3600)
		if isNotNull(row['location']) and isNotNull(row['city']) and str(row['city']) in str(row['location']):
			tue_pop['loc_match'][row_index] = 1

	tue_pop[['user','event','invited_x','interested','not_interested','birthyear','gender','friends_yes','start_hour','time_diff','loc_match','yes','maybe','invited_y','no','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20','c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30','c_31','c_32','c_33','c_34','c_35','c_36','c_37','c_38','c_39','c_40','c_41','c_42','c_43','c_44','c_45','c_46','c_47','c_48','c_49','c_50','c_51','c_52','c_53','c_54','c_55','c_56','c_57','c_58','c_59','c_60','c_61','c_62','c_63','c_64','c_65','c_66','c_67','c_68','c_69','c_70','c_71','c_72','c_73','c_74','c_75','c_76','c_77','c_78','c_79','c_80','c_81','c_82','c_83','c_84','c_85','c_86','c_87','c_88','c_89','c_90','c_91','c_92','c_93','c_94','c_95','c_96','c_97','c_98','c_99','c_100','c_other']].to_csv('train_all.csv', index=False)
	select = tue_pop[['invited_x','birthyear','gender','friends_yes','start_hour','time_diff','loc_match','yes','maybe','invited_y','no','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20','c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30','c_31','c_32','c_33','c_34','c_35','c_36','c_37','c_38','c_39','c_40','c_41','c_42','c_43','c_44','c_45','c_46','c_47','c_48','c_49','c_50','c_51','c_52','c_53','c_54','c_55','c_56','c_57','c_58','c_59','c_60','c_61','c_62','c_63','c_64','c_65','c_66','c_67','c_68','c_69','c_70','c_71','c_72','c_73','c_74','c_75','c_76','c_77','c_78','c_79','c_80','c_81','c_82','c_83','c_84','c_85','c_86','c_87','c_88','c_89','c_90','c_91','c_92','c_93','c_94','c_95','c_96','c_97','c_98','c_99','c_100','c_other','class']]
	for col in select.columns:
		select[col]=select[col].astype(np.float64)
		select[col]=select[col]/max(select[col])

	
	#clf = svm.SVC(gamma=0.001, C=100.)
	clf = tree.DecisionTreeClassifier() 
	clf.fit(select.values[:,:-1], select.values[:,-1])
	pred = clf.predict(select.values[:,:-1])
	target_names = ['class 0', 'class 1']
	result = classification_report(select.values[:,-1], pred, target_names)
	print result

	#select.to_csv('train_feature.csv', index=False)


if __name__=="__main__":
	main()
