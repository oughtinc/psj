import re
import numexpr as ne
import numpy as np


class FermiQuestion:
    """Creates Fermi question objects that we can use later."""

    def __init__(self, length, question, numerical, estimation_difficulty,
                 quantity_ids, categories, quantity_strings):
        self.length = length
        self.question = question
        self.numerical = numerical
        self.estimation_difficulty = estimation_difficulty
        self.quantity_ids = quantity_ids
        self.categories = categories
        self.quantity_strings = quantity_strings
        
        self.features = {}

        self.comparator = '<'
        self.left_val = None
        self.right_val = None
        self.answer = None
        self.evaluate()

    def get_question_quantities(self, question=None):
        """Extracts a list of estimation quantities from a question."""
        if question is None:
            question = self.question
        pattern = r"([A-z]+-?[A-z]*[ \)])+"
        matches = re.finditer(pattern, question)
        return [m.group(0).replace(')', '').strip() for m in matches]

    def get_categories(self):
        return self.categories

    def evaluate(self):
        comparator_idx = self.numerical.index(self.comparator)
        left = self.numerical[:comparator_idx].replace(' ', '')
        right = self.numerical[comparator_idx + 1:].replace(' ', '')
        try:
            self.answer = bool(ne.evaluate(self.numerical))
            self.left_val = int(ne.evaluate(left))
            self.right_val = int(ne.evaluate(right))
        except BaseException as e:
            print('Error while evaluating FermiQuestion: {}'.format(e))

    def get_features(self):
        if not any(self.features):
            self.make_features()
        return self.features

    def make_features(self):
        # Ratio is always positive
        ratio = self.features['ratio'] = max(abs(self.left_val), abs(self.right_val)) / max(
            min(abs(self.left_val), abs(self.right_val)), 1)
        self.features['logratio'] = np.log10(ratio)

        self.features['plus'] = str(self.question).count('+')
        self.features['minus'] = str(self.question).count('-')
        self.features['times'] = str(self.question).count('*')

        return self.features

    def to_save_dict(self):
        self.evaluate()  # TODO might throw
        return {
            'question': self.question,
            'numerical': self.numerical,
            'estimation_difficulty': self.estimation_difficulty,
            'quantity_ids': self.quantity_ids,
            'quantity_strings': self.quantity_strings,
            'categories': self.categories,
            'answer': self.answer,
            'length': self.length
        }

    @staticmethod
    def _split_left_right(q):
        """
        TODO - vectorize

        Splits an array of questions into left and right expressions.
        :param Q: a 1-D numpy array of questions
        :return: a 2-D numpy array of (left, right) expressions
        """
        comparator_idx = q.index('<')
        left = q[:comparator_idx].strip()
        right = q[comparator_idx + 1:].strip()

        return [left, right]

    @staticmethod
    def _rl_difference_char_counts(q, char):
        """I don't know how to properly pass an method as a higher order
        function, so can't reuse `_rl_difference`

        :param char: the char to count
        :return: int representing count

        """
        left, right = FermiQuestion._split_left_right(q)
        diff = right.count(char) - left.count(char)
        return diff

    @staticmethod
    def _rl_difference(count_fn, q):
        """Computes the difference of counts for:

            count_fn(right_expression) - count_fn(left_expression)
        for a single question.
        :param count_fn: function that takes string question and returns int
        :return: int representing count

        """
        left, right = FermiQuestion._split_left_right(q)
        diff = count_fn(right) - count_fn(left)
        return diff

    def __repr__(self):
        A = 'Fermi question:\n'
        B = '  Question: {}\n'.format(self.question)
        C = '  Numerical: {}\n'.format(self.numerical)
        D = '  Answer: {}\n'.format(self.answer)
        return A + B + C + D
