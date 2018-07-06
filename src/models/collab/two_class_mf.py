from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

class TwoClassMF:
    def __init__(self):
        """
        Very simple baseline, using the following algorithm:

        For all questions in the train set, assign to `true` or `false` class
        depending on if the average of responses are > 0.5 or < 0.5.

        Then calculate the average of all responses on `true` questions and set
        this as the `average true rating`, similarly for `false` questions.

        For all responses we need to predict in the test set, look up if it has
        been determined to be `true` or `false` by the train set, if so then
        predict the `average true rating`, similarly for if it is `false`.If it
        doesn't appear in the train set, predict the average response over all
        questions.

        """
        self.question_truth_dict = {}
        self.question_count_dict = {}
        self.question_tot_responses = {}
        self.average_true_rating = 0.5
        self.average_false_rating = 0.5
        self.base_pred = 0.5

    def dataloader_extract(self, sample):
        """Get key features out of the dataloader.  We don't need the
        item text or metadata as we don't use them"""
        ratings = pd.Series(np.array(list(sample['rating'])))
        user_ids = pd.Series(sample['user_id'])
        item_ids = pd.Series(sample['item_id'])

        return ratings, user_ids, item_ids
            
    def fit(self, dataset, train_sampler):
        """Runs the fit method which simply works out the average response for
        'true' and 'false' questions, where 'true' questions are those where the
        average rating is greater than 0.5"""
        
        data_loader = DataLoader(dataset, batch_size=len(train_sampler),
                                 sampler=train_sampler)
        sample = iter(data_loader).next()
        ratings, user_ids, item_ids = self.dataloader_extract(sample)
        
        # Initialise
        for _id in item_ids:
            self.question_count_dict[_id] = 0
            self.question_tot_responses[_id] = 0

        # Collate responses
        for rating, user_id, item_id in zip(ratings, user_ids, item_ids):
            self.question_count_dict[item_id] += 1
            self.question_tot_responses[item_id] += rating

        # Assign to `true` or `false` classes
        for index, tot_rating in self.question_tot_responses.items():
            if tot_rating / self.question_count_dict[index] < 0.5:
                self.question_truth_dict[index] = False
            else:
                self.question_truth_dict[index] = True
                # Note that this means that if avg response is 0.5, set to true.

        # Add up totals for the `true` and `false` classes to get the average
        tot_true, tot_false, count_true, count_false = 0, 0, 0, 0 #laplace smoothing
        for item_id, truth in self.question_truth_dict.items():
            if truth:
                tot_true += self.question_tot_responses[item_id]
                count_true += self.question_count_dict[item_id]
            else:
                tot_false += self.question_tot_responses[item_id]
                count_false += self.question_count_dict[item_id]

        self.base_pred = (tot_true + tot_false) / (count_true + count_false)
        
        if tot_true == 0:
            self.average_true_rating = self.base_pred
        else:
            self.average_true_rating = tot_true / count_true
            
        if tot_false == 0:
            self.average_false_rating = self.base_pred
        else:
            self.average_false_rating = tot_false / count_false

    def predict(self, dataset, sampler):
        preds = []
        data_loader = DataLoader(dataset, batch_size=len(dataset), sampler=sampler)
        sample = iter(data_loader).next()
        ratings, user_ids, item_ids = self.dataloader_extract(sample)
        for rating, user_id, item_id in zip(ratings, user_ids, item_ids):
            if item_id in self.question_truth_dict:
                if self.question_truth_dict[item_id]:
                    preds.append(self.average_true_rating)
                else:
                    preds.append(self.average_false_rating)
            else:
                preds.append(self.base_pred)
                                
        return preds
