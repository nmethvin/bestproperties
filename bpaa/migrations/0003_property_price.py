# Generated by Django 4.0.10 on 2023-09-26 02:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bpaa', '0002_alter_property_bike_score_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='property',
            name='price',
            field=models.IntegerField(default=0, verbose_name='Price'),
            preserve_default=False,
        ),
    ]