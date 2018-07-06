import argparse
import os


import pandas as pd

from definitions import FERMI_RAW_DATA_DIR, FERMI_GEN_DATA_DIR
from src.data.utils import write_fermi_to_csv
from src.data.generate_questions.generate_exprs import generate_unique_category_fermi_questions, generate_fermi_questions

parser = argparse.ArgumentParser(description='Generate fermi data')
parser.add_argument('-q', '--num-questions', type=int, default=20000, help='Number of questions to generate')
parser.add_argument('-u', '--num-users', type=int, default=10, help='Number of users to generate')
parser.add_argument('-c', '--partition-categories', action='store_true')
parser.add_argument('-r', '--ratio', type=float, default=0.8, help='Ratio of quantities to numbers')



def main():
    args = parser.parse_args()
    # responses_path = os.path.join(GEN_DATA_DIR, 'fermi-responses.csv')
    # confidences_path = os.path.join(GEN_DATA_DIR, 'fermi-confidences.csv')
    questions_path = os.path.join(FERMI_GEN_DATA_DIR, 'fermi-questions-augmented.csv')

    # Get raw quantities
    quantities_df = pd.read_csv(os.path.join(FERMI_RAW_DATA_DIR, 'fermi-quantities.csv'))

    if args.partition_categories:
        fqs = generate_unique_category_fermi_questions(
            num_questions=args.num_questions,
            quantities=quantities_df,
            ratio=ratio)
    else:
        fqs = generate_fermi_questions(num_questions=args.num_questions, quantities=quantities_df)


    # Generate user responses and confidences
    #responses_df, confidences_df = generate_responses(num_users=args.num_users, questions=fqs)

    # Write responses and confidences to CSV
    # responses_df.to_csv(responses_path)
    # print('Wrote to {}'.format(responses_path))
    # confidences_df.to_csv(confidences_path)
    # print('Wrote to {}'.format(confidences_path))

    # Write questions to CSVs
    write_fermi_to_csv(questions_path, fqs)
    print('Wrote to {}'.format(questions_path))


if __name__ == '__main__':
    main()
