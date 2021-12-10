#
# author: vva12 Vishakha
#

import argparse
import logging

import api
import utils

if __name__ == '__main__':
    # TODO remove defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to file", default="data")
    parser.add_argument("--coordinates", help="coordinates as tuple of floats", default=None)
    parser.add_argument("--location_str", help="location as string", default="vancouver")
    parser.add_argument("--attributes", help="cuisine types sepearated by ','", default="pizza")
    parser.add_argument("--threshold", help="distance from your location", default=15.0)
    args = parser.parse_args()

    logger = utils.create_logger(logging.DEBUG)
    logging.info("Starting the app")
    api.get_similar_business(
        coordinates=args.coordinates,
        location_str=args.location_str,
        attributes=args.attributes,
        path=args.path,
        threshold=float(args.threshold)
    )


