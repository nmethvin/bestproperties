from django.db import models
from shapely.geometry import Point, Polygon
import ast


class Region(models.Model):
    name = models.CharField(max_length=255)
    crime_rating = models.PositiveIntegerField()
    school_rating = models.PositiveIntegerField()
    real_estate_rating = models.PositiveIntegerField()
    demo_rating = models.PositiveIntegerField()
    coordinates = models.TextField()

    def set_coordinates(self, coords):
        # Convert the tuple to a string and store it
        self.coordinates = str(coords)

    def get_coordinates(self):
        # Safely evaluate the string back to a tuple
        return ast.literal_eval(self.coordinates)

    def is_point_inside(self, x, y):
        coords = self.get_coordinates()
        # If there are more lists, they represent holes, but in this case, there are no holes
        polygon = Polygon(coords)
        point = Point(x, y)

        return point.within(polygon)

    @classmethod
    def get_region_for_point(cls, x, y):
        for region in cls.objects.all():
            if region.is_point_inside(x, y):
                return region
        return None

    def __str__(self):
        return self.name


class Property(models.Model):
    region = models.ForeignKey(Region,
                               on_delete=models.SET_NULL,
                               related_name='property',
                               null=True,
                               blank=True)
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
