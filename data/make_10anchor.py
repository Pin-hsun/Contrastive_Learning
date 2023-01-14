"""random sample of 10 "normal" images from the remain dataset (positive pairs) as anchor"""
import pandas as pd
import numpy as np
import random

pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac1min0base.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac5min0base.csv', index_col=None)
pos = pos.iloc[len(neg):] #only use left pos pairs as anchor
pos = pos[(pos['V00WOMKPR']==0) &( pos['V00WOMKPL']==0)] #only choose both knee womac = 0
pos10 = pos.sample(10)
pos10['side'] = pd.Series(random.choices(['R', 'L'], weights=[1, 1], k=len(pos10))).values
pos10['path'] = 'pain_siamese_pos/Processed/TSE/'+pos10['ID'].astype(str)+'_00_'+pos10['side']
pos10 = pos10[['ID','path']]
pos10.to_csv('/home/gloria/projects/siamese-triplet/data/anchor.csv')