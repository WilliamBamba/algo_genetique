from src.builders import build_default_environment
import cProfile

def main():
    password_cracker = build_default_environment(student_id='11507174')
    password_cracker.run(init_population_size=1000, steps=250, interval=10)

if __name__ == '__main__':
    main()
    #cProfile.run('main()')
