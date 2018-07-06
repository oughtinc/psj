import random
import numexpr as ne
import numpy as np
from math import log10, floor
from src.data.generate_questions.question import FermiQuestion


class ExpressionConfig:

    def __init__(self, quantities, ratio, distribution, reuse=True, max_numeric=200):
        self.reuse = reuse  # whether to reuse quantities
        self.paren_prob = 0.25
        self.fill_str = '{} {} {}'
        self.operators = ['*', '-', '+']
        self.p_operators = np.array([0.6, 0.2, 0.2])
        self.numbers = list(map(str, range(1, max_numeric)))
        self.quants = list(quantities['Quantity'])
        self.diffs = list(quantities['Difficulty'])
        self.quantity_ids = list(quantities['quantity_id'])
        self.categories = list(quantities['Category'])
        self.vals = list(map(str, quantities['Value']))
        self.ratio = ratio
        # Set default distribution
        if distribution is None:
            distribution = {1: 0.3, 2: 0.4, 3: 0.15, 4: 0.15, 5: 0, 6: 0}
        self.dist = []
        self.p_dist = []
        for key, value in distribution.items():
            self.dist.append(key)
            self.p_dist.append(value)
        self.p_dist = [p / float(sum(self.p_dist)) for p in self.p_dist]


def generate_expressions(quantities, num_exprs=10, ratio=0.6, distribution=None, reuse=True, max_numeric=200):
    """
    Returns a list of expressions

    Args:
        quantities: Dataframe of Fermi quantities
        num_exprs: Number of expressions to generate
        ratio: Ratio of quantities in expressions (with the rest being numbers)
        distribution: Discrete distribution over expression length
        reuse: True if quantities should be reused
    """
    cfg = ExpressionConfig(quantities, ratio, distribution, reuse, max_numeric)
    expressions = []
    while len(expressions) < num_exprs and cfg.quants:
        expressions.append(generate_expression(cfg))
    return expressions


def generate_expression(cfg):
    """
    Generates one LHS expression.

    Args:
        cfg: Expression config
    """
    l = np.random.choice(cfg.dist, p=cfg.p_dist)
    done = False
    while not done:
        expression = None
        while cfg.quants and (expression is None or expression['length'] < l):
            expression = add_term(expression, l, cfg)
        try:
            ne.evaluate(expression['numerical'])
            done = True
        except:
            pass

    # If we have a question comprised of only numerical quantities,
    # e.g. 54 * 32, then we don't have any quantity strings. We fill in here
    if 'quantity_strings' not in expression.keys():
        expression['quantity_strings'] = ''
    return expression


def add_term(expression, l, cfg):
    """
    Adds the next term to the expression.

    Args:
        expression: Current expression
        l: Number of terms in final expression
        cfg: Expression config
    """
    term = get_next_term(cfg)
    if expression is None:
        expression = term
        expression['length'] = 1
        if term['categories'] != []:
            # If the term isn't a pure numerical term
            expression['quantity_strings'] = [[term['expression'], term['numerical']]]
    else:
        operator = np.random.choice(cfg.operators, p=cfg.p_operators)
        # Choose which term to add first; original if True
        if np.random.choice([True, False]):
            expression['expression'] = cfg.fill_str.format(expression['expression'], operator, term['expression'])
            expression['numerical'] = cfg.fill_str.format(expression['numerical'], operator, term['numerical'])
        else:
            expression['expression'] = cfg.fill_str.format(term['expression'], operator, expression['expression'])
            expression['numerical'] = cfg.fill_str.format(term['numerical'], operator, expression['numerical'])
        expression['estimation_difficulty'] += term['estimation_difficulty']
        expression['quantity_ids'] = list(set(expression['quantity_ids']).union(set(term['quantity_ids'])))
        expression['categories'] = list(set(expression['categories']).union(set(term['categories'])))
        expression['length'] += 1
        if term['categories'] != []:
            # If the term isn't a pure numerical term
            if 'quantity_strings' in expression.keys():
                expression['quantity_strings'].append([term['expression'], term['numerical']])
            else:
                expression['quantity_strings'] = [[term['expression'], term['numerical']]]

    # Add parentheses
    if (expression['length'] > 1 and np.random.uniform() < cfg.paren_prob and expression['length'] < l and cfg.quants):
        expression['expression'] = '({})'.format(expression['expression'])
        expression['numerical'] = '({})'.format(expression['numerical'])
    return expression


def get_next_term(cfg):
    """
    Gets the next term to be added.

    Args:
        cfg: Expression config
    """
    term = {}
    if np.random.choice(['quantity', 'number'], p=[cfg.ratio, 1 - cfg.ratio]) == 'quantity':
        idx = np.random.choice(range(len(cfg.quants)))
        if cfg.reuse:
            term['expression'] = cfg.quants[idx]
            term['numerical'] = cfg.vals[idx]
            term['estimation_difficulty'] = cfg.diffs[idx]
            term['quantity_ids'] = [cfg.quantity_ids[idx]]
            term['categories'] = [cfg.categories[idx]]
        else:
            term['expression'] = cfg.quants.pop(idx)
            term['numerical'] = cfg.vals.pop(idx)
            term['estimation_difficulty'] = cfg.diffs.pop(idx)
            term['quantity_ids'] = [cfg.quantity_ids.pop(idx)]
            term['categories'] = [cfg.categories.pop(idx)]
    else:
        if len(cfg.numbers) != 200:
            # Where we're not using the default uniform sampling over numbers
            idx = int(np.random.lognormal(3, 8) + abs(np.random.normal(0, 50))) + 1
            term['expression'] = str(idx)
            term['numerical'] = str(idx)
            term['estimation_difficulty'] = 0
            term['quantity_ids'] = []
            term['categories'] = []
        else:
            idx = np.random.choice(range(len(cfg.numbers)))
            term['expression'] = str(idx)
            term['numerical'] = str(idx)
            term['estimation_difficulty'] = 0
            term['quantity_ids'] = []
            term['categories'] = []

    return term


def generate_fermi_questions(num_questions,
                             quantities,
                             ratio=0.8,
                             distribution=None,
                             logratio=1,
                             reuse=True,
                             filter_single_number_lhs=True,
                             balanced=True,
                             max_numeric=200):
    """
    Generates a specified number of Fermi questions.

    Args:
        num_questions: Number of questions to generate
        quantities: Dataframe of Fermi quantities
        ratio: Ratio of quantities in LHS expressions
        distribution: Over lengths of LHS expressions
        logratio: Log ratio standard deviation
        reuse: True if quantities should be reused
        balanced: If the data set should contain exactly the same number of true/false
        max_numeric: How large the numeric quantities should go up to
    """
    cfg = ExpressionConfig(quantities, ratio, distribution, reuse, max_numeric)
    fermi_questions = []
    while cfg.quants and len(fermi_questions) < num_questions:
        fermi_questions.append(generate_fermi_question(cfg, logratio, filter_single_number_lhs))

    # Generation mechanism has a tendency to make a slightly imbalanced
    # dataset: if we need perfect balance we Throw away some and resample
    if balanced:
        true_qs = []
        false_qs = []
        for fermi_question in fermi_questions:
            fermi_question.evaluate()
            if fermi_question.answer:
                true_qs.append(fermi_question)
            else:
                false_qs.append(fermi_question)
        if len(false_qs) > num_questions//2:
            false_qs = false_qs[:num_questions//2]
            while len(true_qs) < num_questions//2:
                new_fq = generate_fermi_question(cfg, logratio, filter_single_number_lhs)
                new_fq.evaluate()
                if new_fq.answer:
                    true_qs.append(new_fq)
        # Randomly interleave the two lists and return
        return [x.pop(0) for x in random.sample([true_qs]*len(true_qs)
                                                + [false_qs]*len(false_qs),
                                                len(true_qs) + len(false_qs))]
    else:
        return fermi_questions


def generate_fermi_question(cfg, logratio, filter_single_number_lhs=True):
    """
    Generates one Fermi question.

    Args:
        cfg: Expression config
        logratio: Log ratio standard deviation (for RHS)
        filter_single_number_lhs: Whether to exclude lhs of a single numerical term
        round_bound: For numbers greater than this, we express in standard
                     form and also make sure the rhs is rounded to 3 sig. figures
    """
    done = False
    rhs_limit = 10**15

    while not done:
        lhs = generate_expression(cfg)
        L = ne.evaluate(lhs['numerical'])
        if L > rhs_limit:
            continue

        if filter_single_number_lhs:
            if len(lhs['quantity_ids']) == 0 and lhs['length'] <= 1:
                continue

        # Always sample the rhs from a
        # lognormal with a larger variance the larger the number is

        if L == 0:
            R = 0
            while R == 0: # Loop until we get an R != L
                R = int(np.random.normal(0, 1))
        else:
            R = L
            while R == L: # Loop until we hit an R != L
                # Now we set the variance of the log RHS so that it
                # grows as the quantity gets bigger
                sd = 0.1 + log10(abs(L)) * 0.065 + log10(abs(L))**2 * 0.0042
                R_raw = sample_lognormal(L, sd)
                # Then round to 3 sf
                if R_raw != 0:
                    R = int(round(R_raw, -int(floor(log10(abs(R_raw)))) + 2))
                else:
                    R = 0

        assert R != L

        try:
            R = ne.evaluate(str(R))
            done = True
        except:
            pass


    question = lhs['expression'] + ' < ' + "{:,}".format(int(R))
    numerical = lhs['numerical'] + ' < ' + str(R)
    fermi_question = FermiQuestion(lhs['length'], question, numerical, lhs['estimation_difficulty'],
                                   lhs['quantity_ids'], lhs['categories'], lhs['quantity_strings'])
    return fermi_question

def sample_lognormal(loc, scale):
    """Given a mean and scale, return int(sgn(mean) * abs(mean) *
    10**r, where r is normally distributed. Handle the case where
    the result is the same as the initial value by trying again."""
    r = np.random.normal(loc=0., scale=scale)
    R = int(np.sign(loc) * abs(loc) * 10**r)
    if loc == R:
        return(sample_lognormal(loc, scale))
    return R

def generate_unique_category_fermi_questions(num_questions,
                                             quantities,
                                             ratio=0.8,
                                             distribution=None,
                                             logratio=0.1,
                                             reuse=True,
                                             filter_single_number_lhs=True):
    """
    Generates a specified number of Fermi questions, such that
    the list of questions has quantities with each category only once

    Args:
        num_questions: Number of questions to generate
        quantities: Dataframe of Fermi quantities
        ratio: Ratio of quantities in LHS expressions
        distribution: Over lengths of LHS expressions
        logratio: Log ratio standard deviation
        reuse: True if quantities should be reused
    """

    num_categories = len(quantities['Category'].value_counts())
    excluded_categories = set()
    cfg = ExpressionConfig(quantities, ratio, distribution, reuse)
    fermi_questions = []
    while (cfg.quants
           and len(fermi_questions) < num_questions
           and len(excluded_categories) < num_categories):
        generated_question = generate_fermi_question(cfg, logratio, filter_single_number_lhs)
        generated_categories = generated_question.categories
        excluded_categories = excluded_categories.union(set(generated_categories))
        cfg = ExpressionConfig(quantities[~quantities['Category'].isin(excluded_categories)],
                               ratio, distribution, reuse)
        fermi_questions.append(generated_question)
    return fermi_questions
