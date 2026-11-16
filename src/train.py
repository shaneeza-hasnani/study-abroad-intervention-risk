import argparse

import pandas as pd

from src.modeling import train
from src.synthetic import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV with documented snapshot-time features and outcome")
    parser.add_argument("--no-xgboost", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, parse_dates=["snapshot_date"]) if args.data else generate()
    _, metadata = train(frame, try_xgboost=not args.no_xgboost)
    print(metadata)


if __name__ == "__main__":
    main()


