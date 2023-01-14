import pandas as pd
import numpy as np
#pos=1, neg=0
"""
label == 1 if samples are from the same class and label == 0 otherwise
pain == 1 if womac>= 5, pain == 0 otherwise
"""
pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac1min0base.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac5min0base.csv', index_col=None)
pos = pos.iloc[:len(neg)+1]

pos['label'] = 1
neg['label'] = 0

pos['painL'] = np.where(pos['V00WOMKPL'] >= 5, 1, 0)
pos['painR'] = np.where(pos['V00WOMKPR'] >= 5, 1, 0)
neg['painL'] = np.where(neg['V00WOMKPL'] >= 5, 1, 0)
neg['painR'] = np.where(neg['V00WOMKPR'] >= 5, 1, 0)

# pos = pos[['ID', 'label','painL','painR']].drop_duplicates()
# neg = neg[['ID', 'label','painL','painR']].drop_duplicates()
pos = pos[['ID', 'label','V00WOMKPL','V00WOMKPR']].drop_duplicates()
neg = neg[['ID', 'label','V00WOMKPL','V00WOMKPR']].drop_duplicates()

pos['path1'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_L'
pos['path2'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_R'
neg['path1'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_L'
neg['path2'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_R'
all = pd.concat([pos, neg], ignore_index=True)
all = all.sort_values(by='ID')
all.to_csv('/home/gloria/projects/siamese-triplet/data/womac_pairs_score.csv',index=False)