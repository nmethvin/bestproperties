# Generated by Django 4.0.10 on 2023-09-25 03:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Property',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sq_ft', models.IntegerField(verbose_name='Square Feet')),
                ('beds', models.IntegerField(verbose_name='Beds')),
                ('bath', models.IntegerField(verbose_name='Bath')),
                ('long', models.FloatField(verbose_name='Longitude')),
                ('lat', models.FloatField(verbose_name='Latitude')),
                ('lot_sq_ft', models.IntegerField(verbose_name='Lot Square Feet')),
                ('walk_score', models.FloatField(verbose_name='Walk Score')),
                ('transit_score', models.FloatField(verbose_name='Transit Score')),
                ('bike_score', models.FloatField(verbose_name='Bike Score')),
                ('address', models.CharField(max_length=255, verbose_name='Address')),
            ],
        ),
    ]
