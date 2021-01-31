import unittest
import numpy as np

from src.builders import build_default_environment


class MyTestCase(unittest.TestCase):

    env = build_default_environment(student_id='11609997')
    random_phenotypes = []

    def test_env(self):
        print('----------- phenotype domain --------------')
        print(self.env.phenotype_domain)

        random_phenotypes = self.env.generate_random_phenotypes(20000)
        print('------- first 5 / {} ---------'.format(random_phenotypes.shape[0]))
        print(random_phenotypes[:5])

        random_phenotypes_lens = np.zeros(random_phenotypes.shape, dtype=np.int32)

        for i, phenotypes in enumerate(random_phenotypes):
            random_phenotypes_lens[i] = len(phenotypes)
            self.assertTrue(self.env.domain_min_size <= random_phenotypes_lens[i] <= self.env.domain_max_size)

        print('random_phenotypes_lens mean')
        print(np.mean(random_phenotypes_lens))

        print('random_phenotypes_lens min')
        print(np.min(random_phenotypes_lens))
        self.assertEqual(self.env.domain_min_size, np.min(random_phenotypes_lens))

        print('random_phenotypes_lens max')
        print(np.max(random_phenotypes_lens))
        self.assertEqual(self.env.domain_max_size, np.max(random_phenotypes_lens))

        self.random_phenotypes = random_phenotypes

    def test_agent(self):
        random_phenotypes = self.env.generate_random_phenotypes(20000)

        scores = self.env.check(phenotypes=random_phenotypes)

        print('------- scores 10 / {} ---------'.format(scores.shape[0]))
        print(scores[:10])

        print('scores mean')
        print(np.mean(scores))

        print('scores min')
        print(np.min(scores))

        print('scores max')
        print(scores[np.argmax(scores)])

        print('scores arg max')
        print(random_phenotypes[np.argmax(scores)])

        print()
        print()


if __name__ == '__main__':
    unittest.main()
