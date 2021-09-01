from network import Network
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('runtype', metavar='N', type=str, nargs='?',
                        help='picked run type, allowed values=[train, test]',
                        default="train")
    parser.add_argument('noit',  type=int, nargs='?', default=10,
                        help='number of iterations during training')
    parser.add_argument('--it', help='tels if test should be interactive', action='store_true')
    return parser


ALLOWED_RUNTYPES = ['train', 'test']


if __name__ == "__main__":
    n = Network()
    pars = get_parser()
    args = pars.parse_args()
    runtype = args.runtype
    assert runtype in ALLOWED_RUNTYPES
    if runtype == "train":
        # noit stands for number of iterations
        noit = args.noit
        n.train(noit)
    else:
        interactivity = True
        if not args.it:
            interactivity = False
        n.classify(interactivity)
