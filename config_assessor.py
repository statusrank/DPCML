# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''

from nni.assessor import Assessor, AssessResult
import logging
import numpy as np 
from collections import defaultdict

logger = logging.getLogger('Customized Assessor')

class CustomizedAssessor(Assessor):
    def __init__(self, epoch_num=3, start_up=1, gap=1, higher_is_better=True):
        super(CustomizedAssessor, self).__init__()
        '''
        param:
            epoch_num:
                the number of epochs to determine whether early stop or not.
            start_up:
                 A trial is determined to be stopped or not only after 
                receiving start_step number of reported intermediate results. 
            gap:
                 The gap interval between Assessor judgements. 
                 For example: if gap = 2, start_step = 6, then we will assess the result 
                 when we get 6, 8, 10, 12…intermediate results.
        '''
        if start_up <= 0:
            logger.warning('It\'s recommended to set start_step to a positive number')
        
        self.epoch_num = epoch_num
        self.start_up = start_up
        self.gap = gap 

        self.higher_is_better = higher_is_better
        self.best_metrics = defaultdict(int)
        self.last_judgment_num = dict()

        logger.info('Successfully initials the Customized assessor')

    def extract_scalar_reward(self, value, scalar_key='default'):
        """
        Extract scalar reward from trial result.
        Parameters
        ----------
        value : int, float, dict
            the reported final metric data
        scalar_key : str
            the key name that indicates the numeric number
        Raises
        ------
        RuntimeError
            Incorrect final result: the final result should be float/int,
            or a dict which has a key named "default" whose value is float/int.
        """
        if isinstance(value, (float, int)):
            reward = value
        elif isinstance(value, dict) and scalar_key in value and isinstance(value[scalar_key], (float, int)):
            reward = value[scalar_key]
        else:
            raise RuntimeError('Incorrect final result: the final result should be float/int, ' \
                'or a dict which has a key named "default" whose value is float/int.')
        return reward

    def extract_scalar_history(self, trial_history, scalar_key='default'):
        """
        Extract scalar value from a list of intermediate results.
        Parameters
        ----------
        trial_history : list
            accumulated intermediate results of a trial
        scalar_key : str
            the key name that indicates the numeric number
        Raises
        ------
        RuntimeError
            Incorrect final result: the final result should be float/int,
            or a dict which has a key named "default" whose value is float/int.
        """
        return [self.extract_scalar_reward(ele, scalar_key) for ele in trial_history]

    def assess_trial(self, trial_job_id, trial_history):
        """
        确定是否要停止该 Trial。 必须重载。
        trial_history: 中间结果列表对象。
        返回 AssessResult.Good 或 AssessResult.Bad。
        """
        curr_step = len(trial_history)
        if curr_step < self.start_up:
            return AssessResult.Good

        self.best_metrics.setdefault(trial_job_id, 0)

        best_metrics = self.best_metrics[trial_job_id]

        trial_history = self.extract_scalar_history(trial_history)
        trial_history = trial_history[-self.epoch_num: ] if len(trial_history) >= self.epoch_num else trial_history
        # print(trial_history)
        if trial_job_id in self.last_judgment_num.keys() and curr_step - self.last_judgment_num[trial_job_id] < self.gap:
            return AssessResult.Good
        self.last_judgment_num[trial_job_id] = curr_step

        if self.higher_is_better:
            best_history = max(trial_history)
            whether_update_best = True if best_metrics <= best_history else False
            sorted_ids = list(np.argsort(-best_history))
            whether_increased = False if (sorted_ids == [i for i in range(len(trial_history))]) else True
            self.best_metrics[trial_job_id] = max(best_history, best_metrics)
        else:
            best_history = min(trial_history)
            whether_update_best = True if best_metrics >= best_history else False
            sorted_ids = list(np.argsort(best_history))
            whether_increased = False if (sorted_ids == [i for i in range(len(trial_history))]) else True
            self.best_metrics[trial_job_id] = min(best_history, best_metrics)

        return AssessResult.Good if whether_update_best else AssessResult.Bad