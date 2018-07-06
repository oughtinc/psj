from src.data.fermi_question import FermiQuestion

def test_get_question_quantities():
    # get quantities test
    my_fermi_question = FermiQuestion(1, '1<2', '1<2', 0, [], [])
    qs = [
        'number of days in non-leap year < 62 + 97 - 61 - 36 - (number of degrees in a right angle + 22)',
        '(number of days in non-leap year) < 62 + 97 - 61 - 36 - (number of degrees in a right angle + 22)',
        '42 + number of days in non-leap year < 62 + 97 - 61 - 36 - (number of degrees in a right angle + 22)',
        '(42 + 36) + number of days in non-leap year < 62 + 97 - 61 - 36 - (number of degrees in a right angle + 22)',
        '(42 + 36) + number of days in non-leap year * 32 < '
        '62 + 97 - 61 - 36 - (number of degrees in a right angle + 22)',
    ]
    actual_quantities = ['number of days in non-leap year', 'number of degrees in a right angle']

    for q in qs:
        parsed_quantities = my_fermi_question.get_question_quantities(q)
        assert (actual_quantities == parsed_quantities)


