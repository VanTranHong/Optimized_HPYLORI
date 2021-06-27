import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
import uni_multiStats as stats
import pickle


###########################         READ DATA           ########
data = pd.read_csv('./data/info.csv',skipinitialspace=True, header = 0)

############################        GENERATING STATISTICS   #####################
### specifying numericals and nominals and target in this dataset
numericals = []
nominals = data.columns 
target="HPYLORI"

data1, nominal_groups = dp.modify_data(data,numericals,nominals, target)

stats.gen_stats(data1, target, nominal_groups)










##########################      GENERATING THE RUNS     #################
n_seed = 5
splits =10
n_features = data1.shape[1]-1
runs = stats.runSKFold(n_seed,splits,data=data1,target=target)
with open("./RUNS/runs.txt","wb") as  fp:
    pickle.dump(runs,fp)
    
#########################       EXECUTE FEATURE SELECTION FOR RANKING BASED      ########
## infogain ####
rsr.execute_feature_selection(runs, ['infogain_20'], n_features,n_seed,splits)
###  reliefF  #####
rsr.execute_feature_selection(runs, ['reliefF_20'], n_features,n_seed,splits)

###  cfs  #####
rsr.execute_feature_selection(runs, ['cfs_0'], n_features,n_seed,splits)

### mrmr   ####
rsr.execute_feature_selection(runs, ['mrmr_0'], n_features,n_seed,splits)






















     































































































































# losreg.fit(X=x_train, y=y_train)
# predictions = linreg.predict(X=x_test)
# error = predictions-y_test
# rmse = np.sqrt((np.sum(error**2)/len(x_test)))
# coefs = linreg.coef_
# features = x_train.columns
# '''


# '''
# #regularization
# alphas = np.linspace(0.0001, 1,100)
# rmse_list = []
# best_alpha = 0

# for a in alphas:
#     lasso = Lasso(fit_intercept = True, alpha = a, max_iter= 10000 )

#     kf = KFold(n_splits=10)
#     xval_err =0




#     for train_index, validation_index in kf.split(x_train):

#         lasso.fit(x_train.loc[train_index,:], y_train[train_index])

#         p = lasso.predict(x_train.iloc[validation_index,:])
#         err = p-y_train[validation_index]
#         xval_err = xval_err+np.dot(err,err)
#         rmse_10cv = np.sqrt(xval_err/len(x_train))
#         rmse_list.append(rmse_10cv)
#         best_alpha = alphas[rmse_list==min(rmse_list)]


# #using the alpha calculated to calculate accuracy of the test
# lasso = Lasso(fit_intercept = True, alpha = best_alpha)
# lasso.fit(x_train, y_train)
# predictionsOnTestdata_lasso = lasso.predict(x_test)
# predictionErrorOnTestData_lasso = predictionErrorOnTestData_lasso-y_test
# rmse_lasso
