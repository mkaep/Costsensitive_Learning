# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:24:38 2020

@author: Manuel Camargo

Modified by Martin Kaeppel (remove all parts that are not necessary for predict next activity
"""
from approach.lstm import next_event_samples_creator as nesc
from approach.lstm import next_event_predictor as nep


class SamplesCreator:
    def create(self, predictor, activity):
        sampler = self._get_samples_creator(activity)
        predictor.sampling(sampler)

    def _get_samples_creator(self, activity):
        if activity == 'predict_next':
            return nesc.NextEventSamplesCreator()
        else:
            raise ValueError(activity)


class PredictionTasksExecutioner:
    def predict(self, predictor, activity):
        executioner = self._get_predictor(activity)
        predictor.predict(executioner)

    def _get_predictor(self, activity):
        if activity == 'predict_next':
            return nep.NextEventPredictor()
        else:
            raise ValueError(activity)
