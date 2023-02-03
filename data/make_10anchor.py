"""random sample of 10 "normal" images from the remain dataset (positive pairs) as anchor"""
import pandas as pd
import numpy as np
import random

pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac3up_ver00.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac0down_ver00.csv', index_col=None)
neg = neg.iloc[len(pos):] #only use left neg pairs as anchor
neg = neg[(neg['V00WOMKPR']==0) &(neg['V00WOMKPL']==0)] #only choose both knee womac = 0
neg10 = neg.sample(10)
neg10['side'] = pd.Series(random.choices(['R', 'L'], weights=[1, 1], k=len(neg10))).values
neg10['path'] = 'pain_siamese_pos/Processed/TSE/'+neg10['ID'].astype(str)+'_00_'+neg10['side']
neg10 = neg10[['ID','path']]
neg10.to_csv('/home/gloria/projects/siamese-triplet/data/anchor.csv')