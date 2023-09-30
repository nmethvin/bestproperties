import argparse
from scripts.get_property_data import get_property_data, add_walk_scores, calc_distance, get_location_data


def main():
    """
    Parser for running all sports data from scripts
    python get_sports_data.py --help for help
    """
    parser = argparse.ArgumentParser(
        description="direct functions to be performed")
    parser.add_argument("--sc", nargs="*", help="scripts to run")
    args = parser.parse_args()
    if args.sc:
        sc = args.sc
    if 'get_property_data' in sc:
        get_property_data()
    elif 'add_walk_scores' in sc:
        add_walk_scores()
    elif 'calc_distance' in sc:
        calc_distance()
    elif 'get_location_data' in sc:
        get_location_data()
    else:
        from scripts.build_model import build_model, run_model
        if 'build_model' in sc:
            build_model()
        elif 'run_model' in sc:
            run_model()


if __name__ == "__main__":
    main()
