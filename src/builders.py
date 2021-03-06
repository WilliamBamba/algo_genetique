from src.models import PasswordCracker


CHANCE_TO_MUTATE_CHAR = 0.8
CHANCE_TO_MUTATE_SIZE = 0.8
CHANCE_TO_SWAP = 0.8
GRADED_RETAIN_PERCENT = 0.3
CHANCE_RETAIN_NONGRATED = 0.05



def build_default_environment(student_id: str = '11507174') -> PasswordCracker:
    numbers = [str(i) for i in range(10)]
    capital_lettres = [chr(i) for i in range(65, 91, 1)]
    phenotype_domain = numbers + capital_lettres
    params = [CHANCE_TO_MUTATE_CHAR,CHANCE_TO_MUTATE_SIZE,CHANCE_TO_SWAP,GRADED_RETAIN_PERCENT,CHANCE_RETAIN_NONGRATED]
    return PasswordCracker(student_id=student_id, possible_values=phenotype_domain, pass_min_size=12, pass_max_size=18,parametres=params)

