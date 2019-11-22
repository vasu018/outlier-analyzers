import pandas as pd
import json
from pylab import *
import matplotlib.pyplot as plt

from numpy import ma
import plotly
import plotly.plotly as py

import random
import pickle

import sys
from colorama import Fore, Back, Style

# Sankey Diagram Function to plot the diagram
def genSankey(df,cat_cols=[],value_cols='',title='Bug Type Plot'):
	
	labelList = []
	for catCol in cat_cols:
		labelListTemp =  list(set(df[catCol].values))
		labelList = labelList + labelListTemp
		
	# remove duplicates from labelList
	labelList = list(dict.fromkeys(labelList))
		
	# transform df into a source-target pair
	for i in range(len(cat_cols)-1):
		if i==0:
			sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
			sourceTargetDf.columns = ['source','target','count']
		else:
			tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
			tempDf.columns = ['source','target','count']
			sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
		sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
		
	# add index for source-target pair
	sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
	sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
	
	# creating the sankey diagram
	data = dict(
		type='sankey',
		node = dict(
		  pad = 15,
		  thickness = 20,
		  line = dict(
			color = "black",
			width = 0.5
		  ),
		  label = labelList,
		  color = df['node_color']
		),
		link = dict(
		  source = sourceTargetDf['sourceID'],
		  target = sourceTargetDf['targetID'],
		  value = sourceTargetDf['count']
		)
	  )
	
	layout =  dict(
		title = title,
		font = dict(
		  size = 15
		)
	)
	   
	fig = dict(data=[data], layout=layout)
	return fig

# Function to create the dataframe as an input for Bug-Type Diagram
def addNamedStructure(globalDF, d_mapping, countDict, namedStructure):
	try:
		df = pd.read_json('outlier_'+namedStructure+'.json', orient='split')
	except:
		print("Named Structure {} not available.".format(namedStructure))
		sys.exit()

	uniqueDeviantProperties = {}
	for i in range(len(df)):
		for prop in df.deviant_properties[i]:
			if prop[0] not in uniqueDeviantProperties:
				uniqueDeviantProperties[prop[0]] = 0
			uniqueDeviantProperties[prop[0]] += 1
	d_namedStructure = {}
	d_namedStructure[namedStructure] = []
	for prop, count in uniqueDeviantProperties.items():
		d_namedStructure[namedStructure].append(prop)
		countDict[prop] = count
	df_net = pd.DataFrame(list(d_namedStructure.items()), columns=['col1', 'col2'])
	df_net = df_net.explode('col2')
	df_net = df_net.reset_index(drop=True)
	
	with open('globalReference.pickle', 'rb') as handle:
		d = pickle.load(handle)
	
	for prop in d_namedStructure[namedStructure]:
		if prop in d:
			d_mapping[prop] = d[prop]
		else:
			print("'{}' Property not in Global File. Add it manually.".format(prop))
	
	globalDF = pd.concat([globalDF, df_net])
	
	return globalDF, d_mapping, countDict


def startPlot():

	globalDF = pd.DataFrame()
	d_mapping = {}
	countDict = {}

	action_selected = 0
	while action_selected < 3:
		
		print()
		print("==========================")
		print("Choose Action:\n")
		print("1 => Add namedStructure")
		print("2 => Plot Diagram")
		print("3 => Exit")
		print("==========================")

		action_selected = input("\nAction => ")
		action_selected = int(action_selected)

		if action_selected == 1:
			print(Fore.RED, end='')
			print("*Add nameStructure Selected*")
			print(Style.RESET_ALL, end='')
			print(Fore.BLUE, end='')
			print("\nEnter all the namedStructure names that you want to plot (comma separated)")
			print(Style.RESET_ALL, end='')
			namedStructure_name = input("=> ")
			print()
			namedStructureNames_arr = [x.strip() for x in namedStructure_name.split(',')]
			for namedStructure_name in namedStructureNames_arr:
				globalDF, d_mapping, countDict = addNamedStructure(globalDF, d_mapping, countDict, namedStructure_name)

		elif action_selected == 2:
			# Creating one dataframe with all the mappings for Bug Type Plot
			df_plot = pd.DataFrame(list(d_mapping.items()), columns=['col1', 'col2'])
			countArr = []
			for prop in df_plot['col1']:
				countArr.append(countDict[prop])
			df_plot['count'] = countArr
			df_plot = df_plot.explode('col2')
			df_plot = df_plot.reset_index(drop=True)
			df_com = pd.merge(df_plot, globalDF, left_on='col1', right_on='col2')
			df_com = df_com.drop(['col2_y'], axis=1)
			df_com.columns = ['property', 'bugType', 'count', 'namedStructure']
			color_arr = []
			for _ in range(len(df_com)):
				r = lambda: random.randint(64,255)
				color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
				color_arr.append(color)
			df_com['node_color'] = color_arr

			# Calling the getSankey function to plot the Bug-Type Diagram
			fig = genSankey(df_com, cat_cols=['namedStructure', 'property', 'bugType'], value_cols='count',title='Types of Bugs')
			plotly.offline.plot(fig, validate=False)

			sys.exit()


startPlot()
