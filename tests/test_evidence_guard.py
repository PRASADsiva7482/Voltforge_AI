import unittest
from engine.evidence_guard import specification_guidance, clarification_without_context


def sources(*texts):
    return [{'snippet': text, 'citationId': f'S{i+1}', 'sourceId': f'doc-{i}'} for i, text in enumerate(texts)]


class EvidenceGuardTest(unittest.TestCase):
    def test_missing_artifacts_get_clarification_not_invented_results(self):
        for question in ['Does my circuit work?', 'Can you explain this schematic?', 'Check that wiring please']:
            self.assertIn('diagram', clarification_without_context(question))
        self.assertIn('exact error', clarification_without_context('Why does my firmware show an error?'))
        for question in ['Hello', 'Explain what a resistor does', 'My code prints error E42',
                         'Why does this circuit use 5 V?', 'My code says: missing semicolon']:
            self.assertIsNone(clarification_without_context(question))
    def test_missing_dimension_never_substitutes_a_different_unit(self):
        for dimension, text in [('current', 'Supply is 3.3 V.'), ('voltage', 'Output is 8 mA.'),
                                ('frequency', 'Supply is 5 V.'), ('resistance', 'Output is 8 mA.')]:
            result = specification_guidance(f'What is the maximum {dimension} of XYZ?', sources(text))
            self.assertIn('do not specify', result)
            self.assertIn(dimension, result)
            self.assertIn('datasheet', result)

    def test_conflict_has_actual_source_values_and_requests_authoritative_conditions(self):
        result = specification_guidance('What is the maximum supply voltage of ABC revision B?',
            sources('ABC revision B maximum supply voltage: 2.5 V. Reference A.',
                    'ABC revision B maximum supply voltage: 4.8 V. Reference B.'))
        self.assertIn('conflict', result)
        self.assertIn('2.5 V [S1]', result)
        self.assertIn('4.8 V [S2]', result)
        self.assertIn('authoritative', result)

    def test_different_revisions_and_conditions_are_not_false_conflicts(self):
        for text in ['ABC revision C maximum voltage: 5 V.',
                     'ABC revision B maximum voltage at 80 C: 5 V.',
                     'ABC revision B maximum voltage: 5 V at 80 C.']:
            self.assertIsNone(specification_guidance('What is the maximum voltage of ABC revision B?',
                sources('ABC revision B maximum voltage: 3.3 V.', text)))

    def test_equivalent_units_do_not_conflict(self):
        self.assertIsNone(specification_guidance('What is the maximum voltage of ABC?',
            sources('ABC maximum voltage: 3.3 V.', 'ABC maximum voltage: 3300 mV.')))

    def test_unrelated_labels_and_one_source_do_not_trigger_conflict(self):
        self.assertIsNone(specification_guidance('What is the maximum voltage of XYZ?',
            sources('ABC maximum voltage: 3.3 V.', 'ABC maximum voltage: 5 V.')))
        self.assertIsNone(specification_guidance('What is the maximum voltage of ABC?',
            sources('ABC maximum voltage: 3.3 V. ABC maximum voltage: 5 V.')))

    def test_explanations_calculations_multi_dimension_and_written_units_defer(self):
        for question in ['Explain current limits', 'What is the difference between voltage and current?',
                         'What maximum current can I calculate from voltage and resistance?',
                         'What is current?', 'Calculate the output current']:
            self.assertIsNone(specification_guidance(question, sources('Supply is 5 V.')))
        self.assertIsNone(specification_guidance('What is the maximum current?', sources('The maximum current is two amps.')))
        self.assertIsNone(specification_guidance('What is the maximum current?', sources('The maximum current is ten milliamperes.')))
        self.assertIsNone(specification_guidance('What is the maximum frequency?', sources('The maximum frequency is four megahertz.')))
        self.assertIsNone(specification_guidance('What is the maximum current?', sources('The maximum current is 10 mA.')))
        self.assertIsNone(specification_guidance('What is the maximum current?', []))


if __name__ == '__main__': unittest.main()
