import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

method = 'SupCon'

if method == 'siamese':
    #pos=0, neg=1
    """
    label == 1 if samples are from the same class and label == 0 otherwise
    pain == 1 if womac>= 5, pain == 0 otherwise
    """
    pos = pd.read_csv('/home/gloria/projects/Contrastive_Learning/data/womac5up_ver00.csv', index_col=None)
    neg = pd.read_csv('/home/gloria/projects/Contrastive_Learning/data/womac0down_ver00.csv', index_col=None)
    neg = neg.iloc[:len(pos)+1]
    dest = 'data/womac_pairs05'

    # set group label
    pos['label'] = 0
    neg['label'] = 1

    # set pain degree label
    # cut off: 0-2->0, 3-5->1, 6+->2
    degree=["high", 'medium', 'low']
    for i in ['L', 'R']:
        col = 'V00WOMKP'+i
        conditions = [pos[col] >= 6, (pos[col] < 6) & (pos[col] > 2), pos[col] <= 2]
        pos['pain'+i] = np.select(conditions, degree, default=np.nan)
    for i in ['L', 'R']:
        col = 'V00WOMKP'+i
        conditions = [neg[col] >= 6, (neg[col] < 6) & (neg[col] > 2), neg[col] <= 2]
        neg['pain'+i] = np.select(conditions, degree, default=np.nan)

    pos = pos[['ID', 'label', 'V00WOMKPL', 'V00WOMKPR', 'painL', 'painR']].drop_duplicates()
    neg = neg[['ID', 'label', 'V00WOMKPL', 'V00WOMKPR', 'painL', 'painR']].drop_duplicates()
    #
    # pos['path1'] = 'pain_siamese_pos3/Processed/TSE/'+pos['ID'].astype(str)+'_00_L'
    # pos['path2'] = 'pain_siamese_pos3/Processed/TSE/'+pos['ID'].astype(str)+'_00_R'
    # neg['path1'] = 'pain_siamese_neg0/Processed/TSE/'+neg['ID'].astype(str)+'_00_L'
    # neg['path2'] = 'pain_siamese_neg0/Processed/TSE/'+neg['ID'].astype(str)+'_00_R'

    all = pd.concat([pos, neg], ignore_index=True)
    all = all[['ID', 'label', 'V00WOMKPL', 'V00WOMKPR', 'painL', 'painR']].drop_duplicates()

if method == 'SupCon':
    data = pd.read_csv('/home/gloria/projects/Contrastive_Learning/data/womac3up_ver00.csv', index_col=None)
    dest = 'data/SupCon_pairs03'
    for i in ['L', 'R']:
        col = 'V00WOMKP'+i
        data['pain' + i] = np.where(data[col] >= 3, 1, 0)
    all = data[['ID', 'V00WOMKPL', 'V00WOMKPR', 'painL', 'painR']].drop_duplicates()

all['path1'] = 'allver00/Processed/TSE/'+all['ID'].astype(str)+'_00_L'
all['path2'] = 'allver00/Processed/TSE/'+all['ID'].astype(str)+'_00_R'
all = all.sort_values(by='ID')
train_index, test_index = train_test_split(list(range(len(all))), test_size=0.3, random_state=42)
train_set = all.iloc[train_index].sort_values(by='ID')
train_set.to_csv(dest+'_train.csv',index=False)
test_set = all.iloc[test_index].sort_values(by='ID')
test_set.to_csv(dest+'_test.csv',index=False)
all.to_csv(dest+'.csv',index=False)

train_set[:20].to_csv(dest+'_part_train.csv',index=False)
test_set[:6].to_csv(dest+'_part_test.csv',index=False)
