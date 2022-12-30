import pandas as pd

pos = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac1min0base.csv', index_col=None)
neg = pd.read_csv('/home/gloria/projects/siamese-triplet/data/womac5min0base.csv', index_col=None)

pos['ID'] = pos['ID'].astype(str) + pos['SIDE'].astype(str)[0][0]
pos = pos[['VER', 'ID', 'SIDE']]
posR = pos[pos['SIDE'] == 'RIGHT'].drop(columns=['SIDE'])
posL = pos[pos['SIDE'] == 'LEFT'].drop(columns=['SIDE'])
