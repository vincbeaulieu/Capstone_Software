
import MachineLearning as ml
import noise_generator

def main():
    test()
    return 0

def test():
    # Comment the test when done
    ml.test()
    noise_amount = noise_generator.generate(1.0) # easy way, no thread

    pass

if __name__ == '__main__':
    main()


