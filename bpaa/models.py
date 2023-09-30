from django.db import models


class Property(models.Model):
    sq_ft = models.IntegerField(verbose_name='Square Feet')
    price = models.IntegerField(verbose_name='Price')
    beds = models.IntegerField(verbose_name='Beds')
    bath = models.IntegerField(verbose_name='Bath')
    garage = models.IntegerField(verbose_name='Garage')
    story = models.IntegerField(verbose_name='Story')
    year = models.IntegerField(verbose_name='Year')
    long = models.FloatField(verbose_name='Longitude')
    lat = models.FloatField(verbose_name='Latitude')
    dist = models.FloatField(verbose_name='Distance', null=True)
    lot_sq_ft = models.IntegerField(verbose_name='Lot Square Feet')
    walk_score = models.FloatField(verbose_name='Walk Score', null=True)
    transit_score = models.FloatField(verbose_name='Transit Score', null=True)
    bike_score = models.FloatField(verbose_name='Bike Score', null=True)
    address = models.CharField(max_length=255, verbose_name='Address')

    def __str__(self):
        return self.address


"""
class Region(models.Model):
    name = models.CharField(max_length=255)
    crime_rating = models.PositiveIntegerField()
    school_rating = models.PositiveIntegerField()
    real_estate_rating = models.PositiveIntegerField()
    demo_rating = models.PositiveIntegerField()

    # For coordinates, you can use a TextField to store them as a JSON-formatted string.
    # However, for more complex geospatial operations, consider using Django's
    # built-in GeoDjango fields and functionalities.
    coordinates = models.TextField()

    def set_coordinates(self, coords_list):
        import json
        self.coordinates = json.dumps(coords_list)

    def get_coordinates(self):
        import json
        return json.loads(self.coordinates)

    def __str__(self):
        return self.name
"""
