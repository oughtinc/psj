import numpy as np
from torch.utils.data import Dataset
from src.models.lang_model.w2v_averager_model import W2vAveragerModel
from sklearn.preprocessing import StandardScaler
import datetime
import pandas as pd
from copy import deepcopy
# import matplotlib.pyplot as plt

"""
Data import functions
"""
def make_dfs(paths):
    df = []
    for path in paths:
        df.append(pd.read_json(path))

    df = pd.concat(df)
    return df

def UUID_to_int(uuid_list):
    map_dict = {}
    for i, uuid in enumerate(uuid_list):
        map_dict[uuid] = i
    return map_dict

def map_id(row, col_name, map_dict):
    val = row[col_name]
    return map_dict[val]

def normalized_seconds(date_series):
    """Given a series of strings in the format
        year-month-day, return a series of floats which
        are normalized (mean 0 and sd 1) unix time"""
    scaler = StandardScaler()
    date_string_list = list(date_series)
    y_m_d_list = [[int(x) for x in date.split('-')] for date in date_string_list]
    unix_times = [datetime.datetime(y, m, d).strftime("%s") for y, m, d in y_m_d_list]
    reshaped_u_ts = np.array(unix_times).reshape(-1, 1).astype('float64')
    np_times = scaler.fit_transform(reshaped_u_ts)
    return pd.DataFrame(np_times, columns=['Date'])

def top_k_one_hot(series, k):
    """Given a pandas series of categorical labels,
        return a one-hot encoding of the top k
        most-frequent labels, with all others under an
        'other' label."""
    series = series.copy()
    counts = series.value_counts()
    mask = series.isin(list(counts.iloc[:k].index))
    series[~mask] = 'Other'
    return pd.get_dummies(series)

def series_to_w2v(series, averager, prefix):
    """Given a pandas series and a W2vAveragerModel object,
        return a dataframe with columns prefix_n
        where n goes up to the size of the returned embedding"""
    w2v_tensor = averager(list(series))
    embed_size = w2v_tensor.data.shape[1]
    col_names = ['{}_{}'.format(prefix, n) for n in range(embed_size)]
    return pd.DataFrame(w2v_tensor.data.numpy(), columns=col_names)


def munge_metadata(df):
    """Given a dataframe with metadata,
    return a one-hot encoded version of that
    dataframe"""
    # One-hot encoding of the parties, states
    new_df = df.copy()
    if 'party' in df.columns:
        parties = pd.get_dummies(new_df['party'])
        parties = parties.rename(index=str, columns={'': 'None'}).reset_index(drop=True)

        states = pd.get_dummies(new_df['state'])
        states = states.rename(index=str, columns={'': 'None'}).reset_index(drop=True)

        #w2v encoding of the contexts
        averager = W2vAveragerModel()
        contexts = new_df['context']
        contexts = series_to_w2v(contexts, averager, 'context').reset_index(drop=True)

        subject = new_df['subject']
        subject = series_to_w2v(subject, averager, 'subject').reset_index(drop=True)

        job = new_df['job']
        job = series_to_w2v(job, averager, 'job').reset_index(drop=True)

        # Do one-hot encoding of speaker such that the top-k speakers are encoded, all else as 'other'
        speakers = new_df['speaker']
        speakers = top_k_one_hot(speakers, 25).reset_index(drop=True)
        #list(speakers.value_counts().iloc[:5].index)

        # some data doesnt have date metadata
        try:
            dates = new_df['date']
            dates = normalized_seconds(dates).reset_index(drop=True)
            fields = [speakers, parties, states, contexts, subject, job, dates]
        except KeyError:
            fields = [speakers, parties, states, contexts, subject, job]

        # Return the index. Unfortunately we have to drop the index names,
        # as otherwise pandas tries to sort the columns, gives up and
        # puts NaNs all over. See https://github.com/pandas-dev/pandas/issues/4588

        return pd.concat(fields, axis=1, ignore_index=True)

    else:
        zero_data = np.zeros(shape=(df.shape[0],1))
        df = pd.DataFrame(zero_data, columns={'': 'None'})
        return df




def data_from_file(results_df, tasks):
    ## Recreate id from index
    tasks['id'] = tasks.index

    ## Add question text
    results_df = results_df.merge(tasks,
                  how='left',
                  left_on='task',
                  right_on='id',
                  suffixes=['_res', '_task'])

    results_df = results_df.dropna(axis=0)

    ## User UUID -> (user, time) -> int id
    results_df['user_time'] = results_df['user'] + results_df['secondsLimit'].astype('str')
    user_mapper = UUID_to_int(results_df['user_time'].unique())
    results_df['user_time_id'] = results_df.apply(map_id, axis=1, args=('user_time', user_mapper))

    ## Question UUID -> int id
    question_mapper = UUID_to_int(results_df['task'].unique())
    results_df['task_id'] = results_df.apply(map_id, axis=1, args=('task', question_mapper))

    ## Remove NaNs, keep selected data
    use_data = results_df[['user_time_id', 'task_id', 'confidence', 'question']].dropna(axis=0)
    vals = use_data.values

    ## Extract metadata from dictionaries and append
    metadata_df = pd.DataFrame(list(results_df['metadata']))
    encoded_metadata = munge_metadata(metadata_df)

    metadata_dict = {}
    text_dict = {}
    for unique_iid, index in zip(np.unique(vals[:, 1], return_index=True)[0],np.unique(vals[:, 1], return_index=True)[1]):
        metadata_dict[unique_iid] = encoded_metadata.iloc[index, :].values
        text_dict[unique_iid] = vals[:, 3][index]


    return ContentDataset(vals[:,0], vals[:,1], vals[:,2], text_dict, metadata_dict)

def sequential_data_from_file(results_df, tasks):
    ## Recreate id from index
    tasks['id'] = tasks.index

    ## Add question text
    results_df = results_df.merge(tasks,
                  how='left',
                  left_on='task',
                  right_on='id',
                  suffixes=['_res', '_task'])

    ## User UUID -> (user, time) -> int id
    user_mapper = UUID_to_int(results_df['user'].unique())
    results_df['user_id'] = results_df.apply(map_id, axis=1, args=('user', user_mapper))

    ## Question UUID -> int id
    question_mapper = UUID_to_int(results_df['task'].unique())
    results_df['task_id'] = results_df.apply(map_id, axis=1, args=('task', question_mapper))

    # how long should sequences be? get it from first response
    seq_len = len(results_df['responseTimesInSeconds'][0])

    user_ids = []
    task_ids = []
    response_seqs = []
    metadata_dict = {}
    texts_dict = {}#['']*len(tasks)

    metadata_list = []
    question_list = []

    for (user, task), data in results_df.groupby(['user_id', 'task_id']):
        sorted_time_df = data.sort_values(by=['secondsLimit'])

        seq = sorted_time_df['confidence'].values
        # keep only fully answered responses
        if not len(seq) == seq_len:
            continue
        if np.isnan(seq).any():
            continue

        user_ids.append(user)
        task_ids.append(task)
        response_seqs.append(seq)
        metadata_list.append(data['metadata'].values[0])
        question_list.append(data['question'].values[0])

    df = pd.DataFrame({'user_id': user_ids,
              'task_id': task_ids,
              'response_seq': response_seqs,
              'metadata': metadata_list,
              'question': question_list})
    df = df.dropna(axis=0)
    use_data = df[['user_id', 'task_id', 'response_seq', 'question']].dropna(axis=0)
    vals = use_data.values

    metadata_df = pd.DataFrame(list(df['metadata']))
    encoded_metadata = munge_metadata(metadata_df)

    metadata_dict = {}
    text_dict = {}
    for unique_iid, index in zip(np.unique(vals[:, 1], return_index=True)[0],np.unique(vals[:, 1], return_index=True)[1]):
        metadata_dict[unique_iid] = encoded_metadata.iloc[index, :].values
        text_dict[unique_iid] = vals[:, 3][index]


    return ContentDataset(vals[:,0], vals[:,1], vals[:,2], text_dict, metadata_dict)


"""
Dataset classes
"""
class ContentDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings, item_text, item_metadata):
        """Wraps some data for iteration by a DataLoader.
        The dataloader has the following attributes:

        user_ids:      a [num_ratings] long list of integers which identify
                       the user giving a particular judgment
                       these are not necessarily contiguous, but
                       can be made so by calling make_ids_contiguous
        item_ids:      similarly for questions
        ratings:       an [num_ratings x timeseries_length] numpy array
                       which holds the judgments at differerent time intervals.
                       All judgments are in the closed interval [0, 1].
                       Note that all the ratings are of the same size (timeseries_length)
                       and having some of the ratings partially not available in the timeseries
                       is achieved by setting the response_mask for that rating.
        item_metadata: an {item_id: metadata} dictionary mapping item ids to
                       the corresponding numpy array with coded metadata information
        item_text:     an {item_id: text} dictionary mapping item ids to
                       the corresponding item texts
        response_mask: A binary [num_ratings x timeseries_length] matrix,
                       where a 1 indicates that the response is available
                       (for testing or training) and a 0 indicates that the
                       response is not available.
        timeseries_length: Cached length of the timeseries
        flat_to_time_dict: A {extended_user_id: (old_user_id x time_index)} map
                           from the new set of users made when flattening the dataset
                           to the old set of users and the time point for that user


        The dataloader has the following functions:

        flatten:     This maps the (user, times) to a new set of responses where
                     each new user only has one time point. This can be reversed
                     by unflatten. Note that the new user ids are not related to
                     the old user ids in any simple way. The mapping between new
                     flat users and (old_user, time) pairs is stored in flat_to_time_dict.
        unflatten:   Reverses flatten.
        right_truncate_to_length: This modifies the dataset to probabilistically
                                  mask the responses to a length given by
                                  its input, [prob_length_1, prob_length_2,
                                  prob_length_3].
        make_ids_contiguous: Some of the methods of creating datasets might provide
                             ids that have gaps. This deals with that condition
                             by changing the ids so that they are in order without
                             gaps, and updating the various places where the indices
                             are stored in the dataloader.

        """

        self.user_ids = user_ids
        self.item_ids = item_ids
        if isinstance(ratings[0], (float, np.float64)): #list of single floats
            self.ratings = np.array(ratings)
            self.timeseries_length = 1
        elif isinstance(ratings, (list, np.ndarray)) and isinstance(ratings[0], (list, np.ndarray)):
            self.timeseries_length = len(ratings[0])
            self.ratings = np.vstack(ratings)
        else: #one np array
            self.timeseries_length =len(ratings[0])
            self.ratings = ratings
        self.item_text = item_text
        self.item_metadata = {iid: np.array(item_metadata[iid]) for
                              iid in item_ids} # put into numpy array
        assert isinstance(item_text, dict)
        self.metadata_size = self.item_metadata[self.item_ids[0]].shape[0]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]

        sample = {'user_id': self.user_ids[idx],
                  'item_id': item_id,
                  'rating': np.array(self.ratings[idx]),
                  'item_metadata': self.item_metadata[item_id],
                  'item_text': self.item_text[item_id]
                  }

        return sample

    def __repr__(self):
        """For use when printing the dateset.
           This just dumps the class contents
        """
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self):
        """For use when printing the dateset.
           This doesn't print the metadata or texts
        """
        return "user_ids: {} \n item_ids: {} \n ratings: {} ".format(
            self.user_ids, self.item_ids, self.ratings)

    def flatten(self):
        new_user_ids = []
        new_item_ids = []
        new_ratings = []

        for i, user_id in enumerate(self.user_ids):
            for t in range(3):
                if not np.isnan(self.ratings[i,t]):
                    new_user_ids.append(3*user_id + t)
                    new_item_ids.append(self.item_ids[i])
                    new_ratings.append(self.ratings[i, t])

        new_user_ids = np.array(new_user_ids)
        new_item_ids = np.array(new_item_ids)
        new_ratings = np.array(new_ratings)

        new_content_dataset = ContentDataset(new_user_ids,
                                             new_item_ids,
                                             new_ratings,
                                             self.item_text,
                                             self.item_metadata)

        return new_content_dataset

    def __exact_mask(self, length, keep_fracs):
        keep_counts = [int(length * kf) for kf in keep_fracs]
        keep_counts[0] = length - sum(keep_counts[1:])

        #build a list of ints describing how many items are kept in each row
        mask = []
        for i, keep_count in enumerate(keep_counts):
            mask += [i] * keep_count
        mask = np.random.permutation(mask)

        mask_2d = np.zeros((length, 3))
        for i, row_keep in enumerate(mask):
            mask_2d[i,:row_keep+1] = 1
        return mask_2d

    def right_truncate_to_length(self, len_fracs):
        """Deletes slower responses according to the probabilities in len_probs.
        The length of len_probs should be the same length as ratings, and must
        sum to one.
        Each entry in len_probs gives the fraction of response vectors that
        will be of that length. For example, if len_probs = [0.7, 0.2, 0.1] then
        the 70\% of response vectors are length one, 20\% are of length two, 
        and 10\% of length three.
        """

        np.testing.assert_almost_equal(sum(len_fracs), 1)
        assert len(len_fracs) == len(self.ratings[0])
        mask = (1-self.__exact_mask(len(self.ratings), len_fracs)).astype(bool)
        self.ratings[mask] = np.nan


    def make_ids_contiguous(self):
        """We may end up in the situation where the user or item ids have
        'gaps', e.g. if the user ids are [0, 1, 1, 5]. This cleans them up."""
        sorted_user_ids = np.unique(self.user_ids)
        sorted_item_ids = np.unique(self.item_ids)
        old_to_new_uid_dict = {user_id: i for i, user_id in enumerate(sorted_user_ids)}
        old_to_new_iid_dict = {item_id: i for i, item_id in enumerate(sorted_item_ids)}

        new_metadata = {old_to_new_iid_dict[iid]: self.item_metadata[iid]
                        for iid in self.item_ids}
        new_item_text = {old_to_new_iid_dict[iid]: self.item_text[iid]
                         for iid in self.item_ids}

        self.item_text = new_item_text
        self.item_metadata = new_metadata

        self.user_ids = [old_to_new_uid_dict[uid] for uid in self.user_ids]
        self.item_ids = [old_to_new_iid_dict[iid] for iid in self.item_ids]

    def train_test_split(self, num_test_questions, p_drop_fast, p_drop_med):
        """Splits the dataset into training and test sets. The ContentDataset
        object on which train_test_split is called becomes the training set, and
        this function then returns the test set.
        """

        question_indices = np.unique(self.item_ids)

        test_questions = np.random.choice(question_indices,
                                          num_test_questions,
                                          replace=False)

        q_and_u = np.stack([self.item_ids, self.user_ids])

        train_user_ids = deepcopy(self.user_ids)
        train_item_ids = deepcopy(self.item_ids)
        train_ratings = deepcopy(self.ratings)

        test_user_ids = []
        test_item_ids = []
        test_ratings = []
        test_metadata = {}
        test_text = {}

        idx_to_delete = []
        for q_idx in test_questions:
            ## Get the users that answered the question
            candidates = np.argwhere(q_and_u[0,:] == q_idx).flatten()

            ## Choose one at random
            test_idx = np.random.choice(candidates)
            test_q, test_u = q_and_u[:, test_idx]

            ## Append data to test set
            test_user_ids.append(test_u)
            test_item_ids.append(test_q)
            test_ratings.append([np.nan, np.nan, self.ratings[test_idx][-1]])
            test_metadata[test_q] = self.item_metadata[test_q]
            test_text[test_q] = self.item_text[test_q]

            ## Delete from training set if not keeping the quicker judgments
            drop_fast = np.random.binomial(1, p_drop_fast)
            drop_med = np.random.binomial(1, p_drop_med)

            ## Could be sped up by storing indices and doing the delete in one
            ## go but we only want to run this function rarely so it's not a
            ## priority to improve
            if drop_fast and drop_med:
                ## item no longer in training set, delete all relevant data
                idx_to_delete.append(test_idx)

            elif drop_fast and not drop_med:
                ## delete only the fast and slow entries from ratings
                train_ratings[test_idx][0] = np.nan
                train_ratings[test_idx][2] = np.nan

            elif drop_med and not drop_fast:
                ## delete only the medium and slow entries from ratings
                train_ratings[test_idx][1] = np.nan
                train_ratings[test_idx][2] = np.nan

            else:
                ## only delete the slow entry from ratings
                train_ratings[test_idx][2] = np.nan

        train_user_ids = np.delete(train_user_ids, idx_to_delete)
        train_item_ids = np.delete(train_item_ids, idx_to_delete)
        train_ratings = np.delete(train_ratings, idx_to_delete, axis=0)

        train_dataset = ContentDataset(train_user_ids,
                                      train_item_ids,
                                      train_ratings,
                                      self.item_text,
                                      self.item_metadata)

        ## Make test dataset and return
        test_dataset = ContentDataset(test_user_ids,
                                      test_item_ids,
                                      test_ratings,
                                      self.item_text,
                                      self.item_metadata) #TODO: should be test_metadata and test_text

        return train_dataset, test_dataset

    # def plot_response_counts(self):
    #     item_ids, response_counts = np.unique(self.item_ids, return_counts=True)
    #
    #     fig, axes = plt.subplots(1)
    #     axes.hist(response_counts)
    #     plt.show()
