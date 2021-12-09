import argparse
import logging
import pandas as pd

# local imports
import api
import utils

if __name__ == '__main__':
    # TODO remove defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to file", default="joined_sample.csv")
    parser.add_argument("--coordinates", help="coordinates as tuple of floats", default=None)
    parser.add_argument("--location_str", help="location as string", default="toronto")
    parser.add_argument("--attributes", help="cuisine types sepearated by ','", default="dosa")
    parser.add_argument("--threshold", help="distance from your location", default=10.0)
    args = parser.parse_args()

    logger = utils.create_logger(logging.DEBUG)
    logging.info("Starting the app")
    # df = pd.read_csv(args.path)
    api.get_similar_business(
        coordinates=args.coordinates,
        location_str=args.location_str,
        attributes=args.attributes,
        df=df,
        threshold=float(args.threshold)
    )


