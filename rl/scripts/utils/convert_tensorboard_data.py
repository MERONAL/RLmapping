from ast import parse
from re import T
from struct import pack
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd


# load data

def main():
    # load the data
    parser = argparse.ArgumentParser(description='Export the tensorboard data')
    parser.add_argument('--in-path', type=str, required=True, help="tensorboard event files")
    parser.add_argument('--ex-path',type=str, required=True, help='location to save the export data')
    args = parser.parse_args()
    event_data = event_accumulator.EventAccumulator(args.in_path)
    event_data.Reload()
    
    keys = event_data.scalars.Keys()
    df = pd.DataFrame()
    event = event_data.Scalars(keys[1])
    steps = [x.step for x in event]
    df['steps'] = pd.DataFrame(steps)
    for key in keys:
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    df.to_csv(args.ex_path)
    print('Tensorboard data exported successfully!')

if __name__ == '__main__':
    main()

