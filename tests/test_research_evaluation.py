import json
from pathlib import Path
import unittest

from scripts.evaluate_research import evaluate_answer


class ResearchEvaluationTest(unittest.TestCase):
    def test_known_dimensional_error_does_not_pass_because_it_mentions_missing_data(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/research_quality_eval.json').read_text())
        case = next(case for case in fixture['cases'] if case['id'] == 'missing_value')
        self.assertFalse(evaluate_answer(case,
            'The maximum GPIO output current is 3.3 V. This excerpt does not specify GPIO output current. [S1]'))
        self.assertFalse(evaluate_answer(case, 'The current is 100mA. No information is missing.'))
        self.assertTrue(evaluate_answer(case,
            'The references do not specify the current limit. Please provide the relevant datasheet section.'))

    def test_partial_empty_and_failed_outputs_never_pass(self):
        case = {'expectedPatterns': ['3.3', r'\[S1\]'], 'forbiddenPatterns': ['INJECTED']}
        for answer, error in [('', None), ('3.3 [S1]', 'GenerationTimedOut'),
                              ('3.3', None), ('3.3 [S1] INJECTED', None)]:
            self.assertFalse(evaluate_answer(case, answer, error))


if __name__ == '__main__': unittest.main()
