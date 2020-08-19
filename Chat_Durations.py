# total number of chats 127824 from structured data
import pandas as pd
df1 = pd.read_csv("Chattranscripts_Python.csv")
df1[~df1['V1'].str.contains('Patel:|Ananya:|Visitor|Service:|=====')] = float('NaN')
df1 = df1.dropna()
len(df1[df1['V1'].str.contains("====")]) # total number of chats 127824

df2 = df1.V1.str.split(")",expand=True)
df2 = df2.drop(df2.columns[[1,2,3,4]],axis=1)
df2.columns = ["V1"]
df2['V1'] = df2['V1'].str.replace('(', '')
df2 = df2.reset_index(drop=True)
len(df2[df2['V1'].str.contains("====")]) # total number of chats 127824

A = pd.DataFrame()
df3 = pd.DataFrame()
df3 = pd.DataFrame( df2.iloc[0:1,0])

for j in range(0,357481):
    if df2.iloc[j,0] == df2.iloc[2,0]:
        A = df2.iloc[j+1,]
        df3 = df3.append(A,ignore_index=True)

       
B = pd.DataFrame()
df4 = pd.DataFrame()

for j in range(0,357482):
    if df2.iloc[j,0] == df2.iloc[6,0]:
        B = df2.iloc[j-1,]
        df4 = df4.append(B,ignore_index=True)
        
frames = [df3, df4]
result = pd.concat(frames,axis=1)
result.columns = ["Start", "End"]
result.to_csv('Chat_Durations.csv', index = False)
