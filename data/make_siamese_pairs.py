import pandas as pd

pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac1min0base.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac5min0base.csv', index_col=None)
pos['label'] = 'pos'
neg['label'] = 'neg'
pos = pos[['ID', 'label']].drop_duplicates()
neg = neg[['ID', 'label']].drop_duplicates()

pos['path1'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_L'
pos['path2'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_R'
neg['path1'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_L'
neg['path2'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_R'
all = pd.concat([pos, neg], ignore_index=True)
all = all.sort_values(by='ID')
all.to_csv('womac_pairs.csv',index=False)