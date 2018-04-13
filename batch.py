# -*- coding: utf-8 -*-

'''
yar ye is ki directories change ekr dy ya apny pass b same different folders bana lay
d:\results\
d:\results\obj
'''
from ExBaggingClassifier import ExBaggingClassifier
import utils
dir_path='D:/results'
def predict_bagging(X, y, file_name,base_estimator=None, seed=10, test_size=0.3, range_estimator=30, max_samples=0.5, max_features=0.5, methods = ['schulze','kemeny_young']):
    for idx in range(len(methods)):
        measures_results = {}
        output = open('{}/{}_{}.txt'.format(dir_path,file_name,methods[idx]), 'a')
        for n_estimators in range(1,range_estimator+1):
            bagging = ExBaggingClassifier(base_estimator,
                                          n_estimators=n_estimators,
                                          random_state=seed, 
                                          max_samples=max_samples, 
                                          max_features=max_features,
                                          voting_method=methods[idx])
            measures_results[methods[idx]+'_{}'.format(n_estimators)] = utils.tts_estimator(bagging, X, y, test_size=test_size, random_state=seed)
            output.write("{} Estimators [{}]\nResult : {}\n".format(methods[idx],n_estimators,measures_results[methods[idx]+'_{}'.format(n_estimators)]))
        output.close()
        print('Voting method : '+methods[idx])
        utils.save_obj(measures_results,'{}/obj/{}_{}.pkl'.format(dir_path,file_name,methods[idx]))
