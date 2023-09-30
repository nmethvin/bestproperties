from bpaa.models import Property
import requests
import django
import os
import sys
import math
import json


def get_property_data():
    total_listings = add_properties(offset=0)
    offset = 200
    while offset < total_listings:
        _total_listings = add_properties(offset=0)
        offset += 200


def add_properties(offset=0):
    url = "https://us-real-estate.p.rapidapi.com/v3/for-sale"

    querystring = {
        "state_code": "TX",
        "city": "Austin",
        "sort": "newest",
        "offset": str(offset),
        "limit": "200"
    }

    headers = {
        "X-RapidAPI-Key": "bb283b7325msh5f9d1c8c6b91c54p106ad3jsnb53697f3760e",
        "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    data = response.json()
    total_listings = data['data']['home_search']['total']
    for prop in data['data']['home_search']['results']:
        try:
            sq_ft = prop['description']['sqft']
            lot_sq_ft = prop['description']['lot_sqft']
            beds = prop['description']['beds']
            bath = float(prop['description']['baths_consolidated'])
            long = float(prop['location']['address']['coordinate']['lon'])
            lat = float(prop['location']['address']['coordinate']['lat'])
            address = prop['location']['address']['line'] + ', ' + prop[
                'location']['address']['city'] + ', ' + prop['location'][
                    'address']['state_code'] + ' ' + prop['location'][
                        'address']['postal_code']
            year = prop['description']['year_built']
            if prop['description']['garage']:
                garage = prop['description']['garage']
            else:
                garage = 0
            story = prop['description']['stories']
            price = prop['list_price']
        except:
            continue
        if not lot_sq_ft:
            continue
        if not sq_ft:
            continue
        prop = get_or_create_property(sq_ft=sq_ft,
                                      beds=beds,
                                      bath=bath,
                                      long=long,
                                      lat=lat,
                                      lot_sq_ft=lot_sq_ft,
                                      address=address,
                                      year=year,
                                      garage=garage,
                                      story=story,
                                      price=price)
        if not prop.dist:
            calc_prop_distance(prop)

    return total_listings


def add_walk_scores():
    properties = Property.objects.filter(walk_score__isnull=True)
    for prop in properties:
        url = "https://walk-score.p.rapidapi.com/score"

        querystring = {
            "lat": f"{str(prop.lat)}",
            "address": prop.address,
            "wsapikey": "63c05cca85d1e103d7969c5c33d1ac06",
            "lon": f"{str(prop.long)}",
            "format": "json",
            "bike": "1",
            "transit": "1"
        }

        headers = {
            "X-RapidAPI-Key":
            "bb283b7325msh5f9d1c8c6b91c54p106ad3jsnb53697f3760e",
            "X-RapidAPI-Host": "walk-score.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        if 'walkscore' in data:
            prop.walk_score = data['walkscore']
        else:
            prop.walk_score = 0
        if 'transit' not in data or 'score' not in data[
                'transit'] or not data['transit']['score']:
            prop.transit_score = 0
        else:
            prop.transit_score = data['transit']['score']
        if 'bike' not in data or 'score' not in data[
                'bike'] or not data['bike']['score']:
            prop.bike_score = 0
        else:
            prop.bike_score = data['bike']['score']
        prop.save()


def get_or_create_property(sq_ft=0,
                           beds=0,
                           bath=0,
                           long=0,
                           lat=0,
                           lot_sq_ft=0,
                           address='',
                           year=1990,
                           story=1,
                           garage=0,
                           price=0):
    try:
        prop = Property.objects.get(address=address)
        if prop.price != price:
            prop.price = price
            prop.save()
        if prop.year != year:
            prop.year = year
            prop.save()
        if prop.story != story:
            prop.story = story
            prop.save()
        if prop.garage != garage:
            prop.garage = garage
            prop.save()
    except:
        prop = Property.objects.create(sq_ft=sq_ft,
                                       beds=beds,
                                       bath=bath,
                                       long=long,
                                       lat=lat,
                                       lot_sq_ft=lot_sq_ft,
                                       address=address,
                                       year=year,
                                       story=story,
                                       garage=garage,
                                       price=price)
    return prop


def calc_distance():
    properties = Property.objects.all()
    for prop in properties:
        distance = distance_2d(prop.lat, prop.long, 30.271535, -97.739182)
        prop.dist = distance
        prop.save()


def calc_prop_distance(prop):
    distance = distance_2d(prop.lat, prop.long, 30.271535, -97.739182)
    prop.dist = distance
    prop.save()


def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_location_data():
    headers = {
        'Referer':
        'https://map_iframe.neighborhoodscout.com/',
        'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    }

    response = requests.get(
        'https://c7d6v7y5.stackpathcdn.com/polygons/bundled_features/cities/15641.json',
        headers=headers)
    data = response.json()
    for region in data:
        region = json.loads(region)
        breakpoint()
