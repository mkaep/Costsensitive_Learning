"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo

Modified by Martin Kaeppel to fit the requirements
"""
import os
import pandas as pd
from approach.lstm import interfaces as it
from keras.models import load_model
from approach.lstm import sim_evaluator as ev
from evaluator import evaluation_lstm


class ModelPredictor():
    """
    This is the man class encharged of the model training
    """
    def __init__(self, parameters, training_log, ac_index, rl_index):
        self.parameters = parameters
        self.model_name, _ = os.path.splitext(parameters['model_file'])
        self.model = load_model(os.path.join(parameters['folder'],
                                             parameters['model_file']))

        self.log = training_log

        self.ac_index = ac_index
        self.rl_index = rl_index

        self.samples = dict()
        self.predictions = None
        self.run_num = 0

        self.execute_predictive_task()

    def sampling(self, sampler):
        self.samples = sampler.create_samples(self.parameters,
                                              self.log,
                                              self.ac_index,
                                              self.rl_index)


    def execute_predictive_task(self):
        sampler = it.SamplesCreator()
        sampler.create(self, 'predict_next')

        for variant in self.parameters['variants']:
            self.imp = variant['imp']
            self.run_num = 0
            for i in range(0, variant['rep']):
                self.predict_values()
                self.run_num += 1

        # assesment
        evaluator = EvaluateTask()
        evaluator.evaluate(self.parameters, self.predictions)

    def predict_values(self):
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, 'predict_next')
        print(2)

    def predict(self, executioner):
        results = executioner.predict(self.parameters,
                                      self.model,
                                      self.samples,
                                      self.imp)

        results = pd.DataFrame(results)
        results['run_num'] = self.run_num
        results['implementation'] = self.imp
        # Only relevant if run_num > 1 and the repetition number is greater 1
        if self.predictions is None:
            self.predictions = results
        else:
            self.predictions = self.predictions.append(results,
                                                       ignore_index=True)
        print(1)

class EvaluateTask():

    def evaluate(self, parameters, data):
        sampler = self._get_evaluator('predict_next')
        return sampler(data, parameters)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parameters):
        exp_desc = self.clean_parameters(parameters.copy())
        evaluation_lstm.evaluate(data, weight_vector=parameters['weight_vector'])


    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        return exp_desc




