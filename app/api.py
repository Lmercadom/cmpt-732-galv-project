import json
import sys
import logging
import pandas as pd
import numpy as np
from geopy import distance
from geopy.geocoders import Nominatim

from utils import create_logger, get_word_vec
from plot_utils import show_recomm

logger = create_logger()

def get_coordinates(location_str):
    """
    get coordinates from location
    :param location_str: str
    :return: (lat,long) tuple
    """
    # set up the connection
    geolocator = Nominatim(user_agent='custom_app')
    location = geolocator.geocode(location_str)
    if location is None:
        raise ValueError(f"Location {location_str} not found")
    return (location.latitude, location.longitude)


def filter_df(df, coord):
    """
    filter the df to only calculate distance of some businesses
    :rtype: object
    :param df: dataframe with lat, long, business_categories, business_name
    :param coord: tuple
    :return: filter mask pd.Series
    """
    df = df.copy()
    df['near'] = False
    int_lat = int(coord[0])
    int_long = int(coord[1])

    # keep only the businesses that have same whole number coordinates
    # from geopy import distance
    # print(distance.distance((52.0, 21.0), (52.0, 20.0)).km)
    # 68.6774747898976
    #
    # print(distance.distance((52.0, 20.0), (51.0, 20.0)).km)
    # 111.257 which is way too far to be a useful recommendation
    df.business_latitude = df.business_latitude.astype(int)
    df.business_longitude = df.business_longitude.astype(int)

    df.loc[(df.business_latitude == int_lat) | (df.business_longitude == int_long), 'near'] = True

    return df.near


def get_distance(df, coord):
    """
    get distance in kms of the filtered businesses
    :param df: pd.DataFrame with near attribute
    :param coord: tuple
    :return: pd.Series with distance in kms or np.nan
    """
    df = df.copy()
    df['distance'] = np.nan
    df.loc[df.near, 'distance'] = df.loc[df.near, ['business_latitude', 'business_longitude']].\
        apply(lambda x: distance.distance((x[0], x[1]), coord).km, axis=1)
    return df.distance


def get_similar_attributes(user_attributes, all_attributes):
    """

    :param user_attributes: user_attributes list
    :param all_attributes: all_attributes set
    :return: dict with user_attributes as keys and similar attributes from all_attributes as list of values
    """
    # get vecs for user and business attributes

    all_vecs = get_word_vec(set(user_attributes) | set(all_attributes))

    # check if user_attributes are avilable in glove DB
    available_user_attr = set(user_attributes) & set(all_vecs.keys())
    available_all_attr = set(all_attributes) & set(all_vecs.keys())
    if len(available_user_attr) == 0:
        logging.info("User provided attributes not found in Glove Vec DB")
        return None

    all_recs = {}
    for attr1 in available_user_attr:
        attr_similarity_dict = {}
        attr_similarity_list = []
        for attr2 in available_all_attr:
            # get similarity with all available attributes
            attr_similarity_dict[attr2] = get_cosine_similarity(all_vecs[attr1], all_vecs[attr2])
        # sort and filter similar attributes
        if len(attr_similarity_dict) > 5:
            attr_similarity_list = sorted(attr_similarity_dict.items(), key=lambda item: item[1], reverse=True)[:5]
        else:
            attr_similarity_list = sorted(attr_similarity_dict.items(), key=lambda item: item[1], reverse=True)
        # keep only the names and not distances
        attr_similarity_list = [i[0] for i in attr_similarity_list]
        all_recs[attr1] = attr_similarity_list

    return all_recs


def get_cosine_similarity(vec1, vec2):
    """

    :param vec1: np.array
    :param vec2: np.array
    :return: distance: float
    """
    dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return dist


def get_similar_business(coordinates=None,
                         location_str=None,
                         attributes=None,
                         df=None,
                         threshold=10.0):
    """
    Get the nearest business based on location
    and/or attributes

    Parameters:
    Input:
        coordinates: tuple(floats)
        location: str
        attributes: str/comma-delimited
        threshold: float

    Output:
        best_businesses: List of business_ids
    """
    df = df.copy()
    user_attributes = attributes.split(',')
    user_attributes = [i.lower() for i in user_attributes]
    if location_str:
        coordinates = get_coordinates(location_str)
        logging.info(f"Coordinates {coordinates}")
    if coordinates is None and location_str is None:
        raise ValueError("provide at least one of location/coordinates")

    # get the near flag
    df['near'] = filter_df(df, coordinates)
    df['distance'] = get_distance(df, coordinates)
    # get the nearest businesses
    df['business_nearby'] = df.distance <= threshold
    # only the distance condition is checked
    if sum(~df.business_nearby) == len(df):
        raise ValueError(f"No business found within {threshold}kms. Try increasing distance or try new locations")

    # filter restaurants on attributes
    df['final_match'] = False
    df.business_categories = df.business_categories.str.lower()
    # distinct attributes of businesses of our interest (business_nearby is True)
    all_attributes = set(df.loc[df.business_nearby, 'business_categories'].str.split(', ', expand=True).stack())
    attribute_match = sum([atr in user_attributes for atr in all_attributes])

    # case when user attributes match one of business attributes
    if attribute_match:
        logging.info("Attributes match found in nearby businesses")
        df.loc[df.business_nearby, 'final_match'] = df.loc[df.business_nearby, 'business_categories'].\
            str.contains("|".join(all_attributes))
    # user provided attributes do not match with businesses
    else:
        logging.info("Attributes match not found in nearby businesses")
        user_attributes_similar = get_similar_attributes(user_attributes, all_attributes)
        if user_attributes_similar is not None:
            similar_attr = []
            for item in user_attributes_similar.values():
                similar_attr.extend(item)
            similar_attr = set(similar_attr)
            logging.debug(f"Similar attributes based on similarity match {similar_attr}")
            df.loc[df.business_nearby, 'final_match'] = df.loc[df.business_nearby, 'business_categories'].\
                str.contains("|".join(similar_attr))

    # we have a final_match variable
    if df.final_match.any():
        rec_business_id = set(df.loc[df.final_match, 'business_id'])
    elif df.business_nearby.any():
        logging.info("No match found on provided user attributes, default to nearby recommendations")
        rec_business_id = set(df.loc[df.business_nearby, 'business_id'])
    else:
        logging.info("No match found nearby OR with similar attributes. Exiting...")
        sys.exit()

    logging.info("plotting recommendations")
    show_recomm(df.loc[df.business_id.isin(rec_business_id)])
