import pandas as pd
import numpy as np
import os
from definitions import HUMAN_DATA_DIR
from sklearn.preprocessing import StandardScaler
import datetime
from src.models.lang_model.w2v_averager_model import W2vAveragerModel
from src.models.content_aware.dataset import ContentDataset

def get_csvs(task, sparsity):
    tr_df = pd.read_csv(os.path.join(HUMAN_DATA_DIR, task, '{}_{}.csv'.format('train', sparsity)), index_col=0)
    val_df = pd.read_csv(os.path.join(HUMAN_DATA_DIR, task, '{}.csv'.format('valid')), index_col=0)
    ho_df = pd.read_csv(os.path.join(HUMAN_DATA_DIR, task, '{}.csv'.format('test')), index_col=0)
    return tr_df, val_df, ho_df

def csv_to_cd(df, ts, metadata_dict):
    ratings = df[ts]
    keys = set(df.item_id)
    item_text_dict = {i:df.item_text[np.where(df.item_id==i)[0][0]] for i in keys}
    item_metadata = {k:metadata_dict[k] for k in keys}
    cd = ContentDataset(df.user_id.values, df.item_id.values, ratings.values, item_text_dict, item_metadata)
    return cd

def get_content_datasets(task, sparsity, include_metadata=True, given_w2v=None, full_detail=False):
    """Loads a content_dataset from a csv. For politifact, gets the metadata unless
       specified not to. If given_w2v is supplied, this is used instead of loading a 
       fresh w2v object, which can take several seconds. If metadata_detail is not None,
       it (an integer) is used to determine the number of top-k columns used."""
    tr_df, val_df, ho_df = get_csvs(task, sparsity)
    
    if task=='fermi':
        ts = ['15', '75', '240']
        keys = set(pd.concat([tr_df, val_df, ho_df]).item_id)
        metadata_dict = {k:np.array([0]) for k in keys}
    elif task=='politifact':
        ts = ['30', '120', '480']
        metadata_df = pd.concat([tr_df, val_df, ho_df]).astype(str).drop_duplicates('item_id')
        keys = set(tr_df.item_id).union(val_df.item_id).union(ho_df.item_id)
        if include_metadata: 
            metadata_dict = {k:r for k,(_,r) in zip(keys,
                                                    munge_metadata(metadata_df,
                                                                   given_w2v,
                                                                   full_detail=full_detail).iterrows())}            
        else:
            metadata_dict = {k:np.array([0]) for k in keys}

    tr = csv_to_cd(tr_df, ts, metadata_dict)
    val = csv_to_cd(val_df, ts, metadata_dict)
    ho = csv_to_cd(ho_df, ts, metadata_dict)
    
    return tr, val, ho

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


def munge_metadata(df, given_w2v=None, full_detail=True):
    """Given a dataframe with metadata,

    return a one-hot encoded version of that
    dataframe"""
    # One-hot encoding of the parties, states
    new_df = df.copy()
    
    # Can supply the w2v model if we want to speed up 
    if given_w2v is None:
        averager = W2vAveragerModel()
    else:
        averager = W2vAveragerModel()
        
    if 'party' in df.columns:
        parties = pd.get_dummies(new_df['party'])
        parties = parties.rename(index=str, columns={'': 'None'}).reset_index(drop=True)

        states = pd.get_dummies(new_df['state'])
        states = states.rename(index=str, columns={'': 'None'}).reset_index(drop=True)

        # Encoding of the contexts
        contexts = new_df['context']
        if full_detail:
            contexts = pd.get_dummies(contexts)
            contexts = contexts.rename(index=str, columns={'': 'None'}).reset_index(drop=True)            
        else:
            contexts = series_to_w2v(contexts, averager, 'context').reset_index(drop=True)

        subject = new_df['subject']
        if full_detail:
            subject = pd.get_dummies(subject)
            subject = subject.rename(index=str, columns={'': 'None'}).reset_index(drop=True)                        
        else:
            subject = series_to_w2v(subject, averager, 'context').reset_index(drop=True)        
        
        job = new_df['job']
        if full_detail:
            job = pd.get_dummies(job)
            job = job.rename(index=str, columns={'': 'None'}).reset_index(drop=True)                        
        else:
            job = series_to_w2v(job, averager, 'context').reset_index(drop=True)

        # Do one-hot encoding of speaker such that the top-k speakers are encoded, all else as 'other'
        speakers = new_df['speaker']
        if full_detail:
            speakers = pd.get_dummies(speakers)
            speakers = speakers.rename(index=str, columns={'': 'None'}).reset_index(drop=True)                        
        else:
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
