import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([1,1,1,1,1])
AgeGroups_1_0  = np.array([0.946666667,0.18,0.786666667,0.702857143,0.368333333])
AgeGroups_2_0 = np.array([0.886666667,0.32,0.73,0.686285714,0.393333333])
AnyAllr = np.array([0.286666667,0.75,0.95,0.811428571,0.665])
AnyPars3 = np.array([0.893333333,0.55,0.95,0.818857143,0.67])
CookArea = np.array([0.78,0.67,0.973333333,0.818857143,0.603333333])
DEWORM  = np.array([0.96,0.75,0.996666667,0.82,0.713333333])
FamilyGrouped_1_0 = np.array([0.8,0.48,0.773333333,0.757714286,0.493333333])
FamilyGrouped_2_0 = np.array([0.893333333,0.56,0.986666667,0.84,0.68])
GCAT_1  = np.array([0.773333333,0.35,0.946666667,0.761714286,0.54])
GCAT_2 = np.array([0.866666667,0.28,0.57,0.804571429,0.356666667])
GCOW = np.array([	0.753333333,0.11,0.663333333,0.720571429,0.326666667])
GDOG_1_0  = np.array([0.9,0.36,0.233333333,0.294857143,0.483333333])
GDOG_2_0 = np.array([0.886666667,0.17,0.836666667,0.709714286,0.383333333	])
GELEC_1_0  = np.array([0.973333333,0.9,0.973333333,0.986857143,0.915])
GELEC_2_0  = np.array([1,1,1,1,1])
GFLOOR6A_1_0 = np.array([0.913333333,0.32,0.14,0.344571429,0.443333333])
GFLOOR6A_2_0 = np.array([0.926666667,0.62,0.776666667,0.722285714,0.385])
GFLOOR6A_9_0 = np.array([0.946666667,0.6,0.396666667,0.453714286,0.406666667])
GWASTE_1_0  = np.array([0.553333333,0.59,0.97,0.804571429,0.543333333])
GWASTE_2_0  = np.array([0.96,0.47,0.87,0.934285714,0.916666667])
GWASTE_3_0  = np.array([0.706666667,0,0.833333333,0.747428571,0.38	])
HCIGR6A = np.array([0.986666667,0.68,0.92,0.976,0.813333333])
ToiletType_1_0 = np.array([	0.966666667,0.76,0.913333333,0.979428571,0.876666667])
ToiletType_2_0 = np.array([0.906666667,0.29,0.033333333,0.280571429,0.206666667	])
WaterSource_1_0 = np.array([0.833333333,0.19,0.476666667,0.753714286,0.298333333])
WaterSource_2_0 = np.array([0.846666667,0.25,0.076666667,0.324,0.151666667])




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
plt.savefig('SFFS,Accuracy_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
