import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([1,1,1,1,1])
AgeGroups_1_0  = np.array([0.953333333,0.27,0.813333333,0.538333333,0.703428571])
AgeGroups_2_0 = np.array([0.966666667,0.52,0.76,0.565,0.698285714])
AnyAllr = np.array([0.106666667,0.67,0.936666667,0.811666667,0.821142857])
AnyPars3 = np.array([0.973333333,0.64,0.893333333,0.731666667,0.793142857])
CookArea = np.array([0.96,0.88,0.983333333,0.736666667,0.878285714])
DEWORM  = np.array([0.96,0.96,0.996666667,0.845,0.887428571])
FamilyGrouped_1_0 = np.array([0.973333333,0.53,0.81,0.61,0.769142857])
FamilyGrouped_2_0 = np.array([0.96,0.56,0.896666667,0.631666667,0.768571429])
GCAT_1  = np.array([0.94,0.5,0.95,0.646666667,0.808])
GCAT_2 = np.array([0.893333333,0.45,0.533333333,0.506666667,0.744571429])
GCOW = np.array([	0.893333333,0.19,0.673333333,0.475,0.709714286])
GDOG_1_0  = np.array([0.993333333,0.42,0.273333333,0.655,0.373142857])
GDOG_2_0 = np.array([0.96,0.27,0.85,0.52,0.704571429	])
GELEC_1_0  = np.array([0.993333333,0.81,0.833333333,0.875,0.937142857])
GELEC_2_0  = np.array([1,0.99,0.993333333,0.99,0.995428571])
GFLOOR6A_1_0 = np.array([0.98,0.36,0.216666667,0.603333333,0.433142857])
GFLOOR6A_2_0 = np.array([0.873333333,0.63,0.783333333,0.531666667,0.685714286])
GFLOOR6A_9_0 = np.array([0.986666667,0.76,0.513333333,0.598333333,0.627428571])
GWASTE_1_0  = np.array([0.74,0.63,0.96,0.638333333,0.801714286])
GWASTE_2_0  = np.array([1,0.85,0.906666667,0.96,0.956])
GWASTE_3_0  = np.array([0.933333333,0.07,0.833333333,0.503333333,0.770285714	])
HCIGR6A = np.array([1,0.16,0.836666667,0.726666667,0.837142857])
ToiletType_1_0 = np.array([	0.966666667,0.47,0.786666667,0.835,0.842285714])
ToiletType_2_0 = np.array([0.96,0.62,0.256666667,0.575,0.553714286	])
WaterSource_1_0 = np.array([0.973333333,0.38,0.493333333,0.473333333,0.771428571])
WaterSource_2_0 = np.array([0.966666667,0.24,0.39,0.36,0.548])




######## calculate the average
ADD_mean=np.mean(ADD)
AgeGroups_1_0_mean=np.mean(AgeGroups_1_0)
AgeGroups_2_0_mean=np.mean(AgeGroups_2_0)
AnyAllr_mean =np.mean(AnyAllr)
AnyPars3_mean=np.mean(AnyPars3)
CookArea_mean=np.mean(CookArea)
DEWORM_mean=np.mean(DEWORM)
FamilyGrouped_1_0_mean=np.mean(FamilyGrouped_1_0)
FamilyGrouped_2_0_mean=np.mean(FamilyGrouped_2_0)
GCAT_1_mean=np.mean(GCAT_1)
GCAT_2_mean=np.mean(GCAT_2)
GCOW_mean =np.mean(GCOW)
GDOG_1_0_mean=np.mean(GDOG_1_0)
GDOG_2_0_mean=np.mean(GDOG_2_0)
GELEC_1_0_mean=np.mean(GELEC_1_0)
GELEC_2_0_mean=np.mean(GELEC_2_0)
GFLOOR6A_1_0_mean=np.mean(GFLOOR6A_1_0)
GFLOOR6A_2_0_mean=np.mean(GFLOOR6A_2_0)
GFLOOR6A_9_0_mean=np.mean(GFLOOR6A_9_0)
GWASTE_1_0_mean=np.mean(GWASTE_1_0)
GWASTE_2_0_mean=np.mean(GWASTE_2_0)
GWASTE_3_0_mean=np.mean(GWASTE_3_0)
HCIGR6A_mean=np.mean(HCIGR6A)
ToiletType_1_0_mean=np.mean(ToiletType_1_0)
ToiletType_2_0_mean=np.mean(ToiletType_2_0)
WaterSource_1_0_mean=np.mean(WaterSource_1_0)
WaterSource_2_0_mean=np.mean(WaterSource_2_0)





####### calculate the standard deviation

ADD_std = np.std(ADD)/math.sqrt(5)
AgeGroups_1_0_std=np.std(AgeGroups_1_0)/math.sqrt(5)
AgeGroups_2_0_std=np.std(AgeGroups_2_0)/math.sqrt(5)
AnyAllr_std=np.std(AnyAllr)/math.sqrt(5)
AnyPars3_std=np.std(AnyPars3)/math.sqrt(5)
CookArea_std=np.std(CookArea)/math.sqrt(5)
DEWORM_std=np.std(DEWORM)/math.sqrt(5)
FamilyGrouped_1_0_std=np.std(FamilyGrouped_1_0)/math.sqrt(5)
FamilyGrouped_2_0_std=np.std(FamilyGrouped_2_0)/math.sqrt(5)
GCAT_1_std=np.std(GCAT_1)/math.sqrt(5)
GCAT_2_std=np.std(GCAT_2)/math.sqrt(5)
GCOW_std=np.std(GCOW)/math.sqrt(5)
GDOG_1_0_std=np.std(GDOG_1_0)/math.sqrt(5)
GDOG_2_0_std=np.std(GDOG_2_0)/math.sqrt(5)
GELEC_1_0_std=np.std(GELEC_1_0)/math.sqrt(5)
GELEC_2_0_std=np.std(GELEC_2_0)/math.sqrt(5)
GFLOOR6A_1_0_std=np.std(GFLOOR6A_1_0)/math.sqrt(5)
GFLOOR6A_2_0_std=np.std(GFLOOR6A_2_0)/math.sqrt(5)
GFLOOR6A_9_0_std=np.std(GFLOOR6A_9_0)/math.sqrt(5)
GWASTE_1_0_std=np.std(GWASTE_1_0)/math.sqrt(5)
GWASTE_2_0_std=np.std(GWASTE_2_0)/math.sqrt(5)
GWASTE_3_0_std=np.std(GWASTE_3_0)/math.sqrt(5)
HCIGR6A_std=np.std(HCIGR6A)/math.sqrt(5)
ToiletType_1_0_std=np.std(ToiletType_1_0)/math.sqrt(5)
ToiletType_2_0_std=np.std(ToiletType_2_0)/math.sqrt(5)
WaterSource_1_0_std=np.std(WaterSource_1_0)/math.sqrt(5)
WaterSource_2_0_std=np.std(WaterSource_2_0)/math.sqrt(5)




######### create lists for the plot

Features = ['Rf1','Rf2','Rf3','Rf4','Rf5','Rf6','Rf7','Rf8','Rf9','Rf10','Rf11','Rf12','Rf13','Rf14','Rf15','Rf16','Rf17','Rf18','Rf19','Rf20','Rf21','Rf22','Rf23','Rf24','Rf25','Rf26','Rf27']


x_pos = np.arange(len(Features))
CTEs = [ADD_mean,CookArea_mean,HCIGR6A_mean,GCAT_2_mean,GELEC_1_0_mean,GELEC_2_0_mean,DEWORM_mean,GCAT_1_mean,ToiletType_1_0_mean,AnyPars3_mean,AnyAllr_mean,GWASTE_2_0_mean,GDOG_1_0_mean,GCOW_mean,FamilyGrouped_2_0_mean,GWASTE_1_0_mean,AgeGroups_1_0_mean,GFLOOR6A_9_0_mean,ToiletType_2_0_mean,GFLOOR6A_1_0_mean,GWASTE_3_0_mean,WaterSource_1_0_mean,FamilyGrouped_1_0_mean,AgeGroups_2_0_mean,GFLOOR6A_2_0_mean,WaterSource_2_0_mean,GDOG_2_0_mean]
error = [ADD_std,CookArea_std,HCIGR6A_std,GCAT_2_std,GELEC_1_0_std,GELEC_2_0_std,DEWORM_std,GCAT_1_std,ToiletType_1_0_std,AnyPars3_std,AnyAllr_std,GWASTE_2_0_std,GDOG_1_0_std,GCOW_std,FamilyGrouped_2_0_std,GWASTE_1_0_std,AgeGroups_1_0_std,GFLOOR6A_9_0_std,ToiletType_2_0_std,GFLOOR6A_1_0_std,GWASTE_3_0_std,WaterSource_1_0_std,FamilyGrouped_1_0_std,AgeGroups_2_0_std,GFLOOR6A_2_0_std,WaterSource_2_0_std,GDOG_2_0_std]

#########build the plot
#########build the plotpython
fig, ax = plt.subplots()
ax.bar(x_pos,CTEs,yerr = error,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xlabel('Feature', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(Features)

ax.yaxis.grid(True)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('SFFS,F1_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
