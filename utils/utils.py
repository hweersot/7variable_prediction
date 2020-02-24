import argparse

# argparse 라이브러리를 이용해 인자로 들어온 값을 args에 모아 전달
def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')
    args = argparser.parse_args()
    return args
