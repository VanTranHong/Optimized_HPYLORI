import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import math


names = ['Residence','Allergies','Parasites','CookArea','Deworms','Cow','Smoking','CatInside','CatOutside','DogInside','DogOuside','ElecSometimes','ElecNever', 'WoodFloor','MudFloor', 'OtherFloor', 'WastePit', 'WasteField','WasteBurn','6-10Years', '11-15Years','4-5FamSize', '>5Famsize','ToiletPit','ToiletField','WaterWell','WaterNatural']
df = pd.read_csv('/Users/vantran/Desktop//RESEARCH/PLOTS/ranking_based.csv')
row = df.shape[0]
cols = df.shape[1]


average = []
std = []
for i in range(row):
    rw = df.iloc[i].tolist()
    rw = [i/10 for i in rw]
    average.append(np.mean(rw))
    std.append(np.std(rw)/math.sqrt(cols))
    
new_ave =[]
new_std =[]
Features = list(range(1,28))
for i in Features:
    new_ave.append(average[i-1])
    new_std.append(std[i-1])
x_pos = np.arange(len(Features))



fig, ax = plt.subplots()
ax.bar(x_pos,new_ave,yerr = new_std,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xlabel('Risk factor', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
# ax.grid(False)

ax.yaxis.grid(False)

# Save the figure and show
plt.xticks(rotation =78)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('Ranking_based_plot_with_error_bars.png')
plt.show()



    
    # average.append()