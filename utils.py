from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', action='store_true', help='Server')
    parser.add_argument('-t', action='store_true', help='Trainer')
    parser.add_argument('-a', action='store_true', help='Actor')

    return parser.parse_args()