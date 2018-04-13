# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:00:36 2018

@author: Rizwan Yasin
"""

import numpy as np
#import operator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators
from sklearn.ensemble import BaggingClassifier
from sklearn.externals.six.moves import zip
from profile_choice import Profile

def ballot_papers(ballot_box):
    m = [[[chr(ord('A') + i[0]) for i in sorted([(idx, val) for idx, val in enumerate(bp)], reverse=True, key=lambda x:x[1])] for bp in bb] for bb in ballot_box]
    return m

def _parallel_predict_proba(estimators, estimators_features, X, n_classes, voting_method):
	"""Private function used to compute (proba-)predictions within a job."""
	n_samples = X.shape[0]
	p = []
	proba = np.zeros((n_samples, n_classes))
	for estimator, features in zip(estimators, estimators_features):
		if hasattr(estimator, "predict_proba"):
			proba_estimator = estimator.predict_proba(X[:, features])
			#print(proba_estimator[0].sum())
			if n_classes == len(estimator.classes_):
				proba += proba_estimator
				p = p + [proba_estimator]
			else:
				proba[:, estimator.classes_] += \
					proba_estimator[:, range(len(estimator.classes_))]
				tmp = np.zeros((n_samples, n_classes))
				tmp[:, estimator.classes_] = \
					proba_estimator[:, range(len(estimator.classes_))]
				p = p + [tmp]
		else:
			# Resort to voting
			predictions = estimator.predict(X[:, features])
			for i in range(n_samples):
				proba[i, predictions[i]] += 1
	#print(proba)
	methods=['kemeny_young','schulze']
	if voting_method in methods and hasattr(estimator, "predict_proba"):
		ballot_box = [list(i) for i in zip(*p)]
		rank_p = np.zeros((n_samples, n_classes))
		ii=0
		#ballot_box=ballot_papers(ballot_box)
		for ballots in ballot_box:
			profile = Profile.ballot_box(ballots)
			if voting_method == 'kemeny_young':
				rank = profile.kemeny_young()
			else:
				rank = profile.ranking(eval('profile.'+voting_method))
			s_rank = [v for idx,v in sorted(rank, key=lambda x:x[0])]
			rank_p[ii]+= s_rank
			ii = ii + 1
		return rank_p
	else:
		return proba

class ExBaggingClassifier(BaggingClassifier):
	def __init__(self, base_estimator=None,n_estimators=10,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,oob_score=False,warm_start=False,n_jobs=1,random_state=None,verbose=0,voting_method='default'):
		super(BaggingClassifier, self).__init__(base_estimator,n_estimators=n_estimators,max_samples=max_samples,max_features=max_features,bootstrap=bootstrap,bootstrap_features=bootstrap_features,oob_score=oob_score,warm_start=warm_start,n_jobs=n_jobs,random_state=random_state,verbose=verbose)
		self.voting_method = voting_method
	def predict(self, X):
		predicted_probabilitiy = self.predict_proba(X)
		return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)
	def predict_proba(self, X):
		check_is_fitted(self, "classes_")
		# Check data
		X = check_array(X, accept_sparse=['csr', 'csc'])
		if self.n_features_ != X.shape[1]:
			raise ValueError("Number of features of the model must "
							 "match the input. Model n_features is {0} and "
							 "input n_features is {1}."
							 "".format(self.n_features_, X.shape[1]))
		# Parallel loop
		n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,self.n_jobs)
		all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
			delayed(_parallel_predict_proba)(
				self.estimators_[starts[i]:starts[i + 1]],
				self.estimators_features_[starts[i]:starts[i + 1]],
				X,
				self.n_classes_,self.voting_method)
			for i in range(n_jobs))
		# Reduce
		#print('Bag Sum:',all_proba)
		proba = sum(all_proba) / self.n_estimators
		#print(self.n_estimators, 'local probability : ', all_proba,'##AVG##',proba)
		return proba

