
import logging
from operator import itemgetter

import msgpack
import numpy as np


class Perceptron():
    '''
    A simple implementation of the Averaged Perceptron.
    '''

    def __init__(self):
        self.weights = None

    def train(self, train_examples, max_iters):
        '''
        This performs the Averaged Perceptron,
        using a clever implementation trick from Algorithm 7 of
        http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
        to compute the averaged weights efficiently.
        '''
        np.random.seed(1234567890)

        # initialize the dict objects for each action
        self.weights = {}
        weights_avg = {}
        valid_actions = {example['y'] for example in train_examples}
        for action in valid_actions:
            self.weights[action] = {}
            weights_avg[action] = {}

        # logging stuff
        log_interval = 1000
        correctness = []
        c = 1  # counter for weight averaging

        # go through the training set max_iters times
        for i in range(max_iters):
            logging.info("training iteration {}".format(i))
            np.random.shuffle(train_examples)

            for example in train_examples:

                # for each example, make a prediction
                feats = example['x']
                scores = self.compute_scores(feats)
                y_hat = self.find_best_action(scores)

                # adjust weights if incorrect (Perceptron update)
                y = example['y']
                correctness.append(1.0 if y_hat == y else 0.0)
                if y_hat != y:
                    for feat, val in feats.items():
                        self.weights[y_hat][feat] = self.weights[y_hat].get(feat, 0.0) - val
                        self.weights[y][feat] = self.weights[y].get(feat, 0.0) + val
                        weights_avg[y_hat][feat] = weights_avg[y_hat].get(feat, 0.0) - val * c
                        weights_avg[y][feat] = weights_avg[y].get(feat, 0.0) + val * c

                # logging stuff
                if c % log_interval == 0:
                    logging.info('iter {}, processed {} examples total'.format(i, c))
                    logging.info('prop. correct of last {}: {}'.format(log_interval, np.mean(correctness)))
                    correctness.clear()

                c += 1

        # compute the final, averaged weights
        for action, action_weights in self.weights.items():
            for feat, val in action_weights.items():
                action_weights[feat] = val - float(weights_avg[action][feat]) / c

    @staticmethod
    def find_best_action(scores):
        return sorted(scores.items(), key=itemgetter(1), reverse=True)[0][0]

    def load_model(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.weights = msgpack.load(model_file, encoding='utf-8')

    def save_model(self, model_path):
        with open(model_path, 'wb') as model_file:
            msgpack.dump(self.weights, model_file, use_bin_type=True)

    def compute_scores(self, feats):
        '''
        This computes the dot products between the features and the weights
        for each action.
        '''
        scores = {}
        for action, action_weights in self.weights.items():
            scores[action] = 0.0
            for feat, val in feats.items():
                if feat in action_weights:
                    scores[action] += action_weights[feat] * val
        return scores
