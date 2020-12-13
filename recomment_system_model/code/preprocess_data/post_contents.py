import pickle
import pandas as pd
import os
import csv

from CONFIG import *

with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
content_files = open(os.path.join(LINK_DATA,"data_preparation.csv"),"r")
data=csv.reader(content_files)

print(data)
print(len(item_dict))
content_dict=dict()
a=0
for i in data:
    try:
        if i[1]!="" or i[1]!=" ":
            content_dict[i[0]]=i[1]
            a = a+1
    except:
        print(i[0])
print(a)
#print(content_dict)
with open(os.path.join(LINK_DATA,"contens.picke"),"wb") as out_put_file:
    pickle.dump(content_dict,out_put_file)