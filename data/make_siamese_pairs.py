import pandas as pd
#pos=1, neg=0
"""
Contrastive loss
Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
"""
pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac1min0base.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac5min0base.csv', index_col=None)
pos['label'] = 1
neg['label'] = 0
pos = pos[['ID', 'label']].drop_duplicates()
neg = neg[['ID', 'label']].drop_duplicates()

pos['path1'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_L'
pos['path2'] = 'pain_siamese_pos/Processed/TSE/'+pos['ID'].astype(str)+'_00_R'
neg['path1'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_L'
neg['path2'] = 'pain_siamese_neg/Processed/TSE/'+neg['ID'].astype(str)+'_00_R'
all = pd.concat([pos, neg], ignore_index=True)
all = all.sort_values(by='ID')
all.to_csv('womac_pairs.csv',index=False)