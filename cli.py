import argparse, sys
from pkg_resources import resource_filename
from .model import load_model, predict

KAGGLE_ID = "cloechapotot"

def main():
    p = argparse.ArgumentParser(prog='inference')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input',  help='Text to classify')
    g.add_argument('--kaggle', action='store_true')
    args = p.parse_args()

    if args.kaggle:
        print(KAGGLE_ID); sys.exit(0)

    # load assets
    mpath = resource_filename('emotion_classifier.assets','best_model.pt')
    vpath = resource_filename('emotion_classifier.assets','vocab.json')
    mdl   = load_model(mpath, vpath)
    print(predict(mdl, args.input))
