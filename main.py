import argparse
from src.get_data import *
from src.clean_data import *
from src.run_analysis import *
from src.visualize_results import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process data')

    # Define command-line arguments
    parser.add_argument('-clean', action='store_true', help='Clean data')
    parser.add_argument('-get', action='store_true', help='Get data')
    parser.add_argument('-analyze', action='store_true', help='Analyze data')
    parser.add_argument('-visualize', action='store_true', help='Visualize data')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.get:
        get_data()

    elif args.clean:
        clean_data()

    elif args.analyze:
        run_analysis()

    elif args.visualize:
        visualize_results()
