import pandas as pd
import csv


def get_user_names(data, removed_cols=('answer', 'task_data', 'user_data')):
    return list(set(data.columns.levels[0]) - set(removed_cols))


def get_times(data):
    user_names = get_user_names(data)
    times = sorted(set(data[user_names].columns.get_level_values(1)))
    return times


def fqs_to_df(fqs):
    """Converts a list of Fermi question objects to a Dataframe."""
    cols = ['question', 'numerical', 'estimation_difficulty', 'quantity_ids', 'categories', 'answer', 'length']
    fqs_df = pd.DataFrame([fq.to_save_dict() for fq in fqs])
    fqs_df = fqs_df[cols]
    return fqs_df


def write_fermi_to_csv(csv_filename, fqs):
    """
    Writes a list of Fermi question objects to csv.
    Args:
        csv_filename: Filename for output csv
        fqs: List of Fermi question objects
    """
    fieldnames = ['question', 'numerical',
                  'estimation_difficulty',
                  'quantity_ids', 'categories',
                  'answer', 'length', 'quantity_strings']
    row_dicts = [fq.to_save_dict() for fq in fqs]
    with open(csv_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row_dict in row_dicts:
            writer.writerow(row_dict)
