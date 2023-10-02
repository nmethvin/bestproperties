from bpaa.models import Property, Region
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_percentage_error, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import numpy
from django.db.models import F


def build_model():
    # Extract data from Django model
    properties = Property.objects.annotate(price_per_sqft=F('price') /
                                           F('sq_ft'))
    total_properties = properties.count()

    # Calculate the indices for the 10% and 90% percentiles
    bottom_10_percent_index = int(total_properties * 0.05)
    top_10_percent_index = int(total_properties * 0.95)

    # Get the thresholds
    properties_ordered = properties.order_by('price_per_sqft')
    bottom_threshold = properties_ordered[
        bottom_10_percent_index].price_per_sqft
    top_threshold = properties_ordered[top_10_percent_index].price_per_sqft
    properties = properties.filter(price_per_sqft__gt=bottom_threshold,
                                   price_per_sqft__lt=top_threshold,
                                   region__isnull=False)
    regions_df = pd.DataFrame(list(Region.objects.all().values()))

    # properties = Property.objects.all().values()
    properties_df = pd.DataFrame(list(properties.values()))
    df = pd.merge(properties_df,
                  regions_df,
                  left_on='region_id',
                  right_on='id',
                  suffixes=('', '_region'))

    # Split the data
    features = df.drop([
        'price', 'address', 'walk_score', 'transit_score', 'bike_score',
        'long', 'lat', 'price_per_sqft', 'name', 'coordinates'
    ],
        axis=1)  # Assuming 'price' is your target variable
    target = df['price']
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42)

    # Normalize or standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1], )),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='rmsprop', loss='mean_absolute_percentage_error')
    model.fit(X_train,
              y_train,
              epochs=500,
              batch_size=32,
              validation_data=(X_test, y_test))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # Predict some values
    predictions = model.predict(X_test)
    # print(predictions)

    mae = median_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)
    print(f"Median Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}")
    print(f"Explained Variance Score: {evs:.2f}")

    model.save("property_price_model.h5")
    joblib.dump(scaler, "scaler.pkl")


def run_model(request,
              max_price='',
              min_price='',
              max_distance='',
              built_after='',
              built_before='',
              max_beds='',
              min_beds='',
              max_bath='',
              min_bath='',
              max_sqft='',
              min_sqft='',
              num_of_results=''):
    properties = Property.objects.annotate(price_per_sqft=F('price') /
                                           F('sq_ft'))
    total_properties = properties.count()

    # Calculate the indices for the 10% and 90% percentiles
    bottom_10_percent_index = int(total_properties * 0.05)
    top_10_percent_index = int(total_properties * 0.95)

    # Get the thresholds
    properties_ordered = properties.order_by('price_per_sqft')
    bottom_threshold = properties_ordered[
        bottom_10_percent_index].price_per_sqft
    top_threshold = properties_ordered[top_10_percent_index].price_per_sqft
    properties = properties.filter(price_per_sqft__gt=bottom_threshold,
                                   price_per_sqft__lt=top_threshold,
                                   region__isnull=False)
    if min_price:  # Check if min_price is not empty
        properties = properties.filter(
            price__gte=float(min_price)
        )  # Filter properties with price greater than or equal to min_price

    if max_price:  # Check if max_price is not empty
        properties = properties.filter(
            price__lte=float(max_price)
        )  # Filter properties with price less than or equal to max_price
    if max_distance:  # Check if min_price is not empty
        properties = properties.filter(
            dist__lte=float(max_distance) * 69
        )  # Filter properties with price greater than or equal to min_price

    if built_after:  # Check if max_price is not empty
        properties = properties.filter(
            year__gt=float(built_after)
        )  # Filter properties with price less than or equal to max_price
    if built_before:  # Check if max_price is not empty
        properties = properties.filter(
            year__lt=float(built_before)
        )  # Filter properties with price less than or equal to max_price
    if max_beds:  # Check if max_price is not empty
        properties = properties.filter(
            beds__lte=float(max_beds)
        )  # Filter properties with price less than or equal to max_price
    if min_beds:  # Check if max_price is not empty
        properties = properties.filter(
            beds__gte=float(max_beds)
        )  # Filter properties with price less than or equal to max_price
    if max_bath:  # Check if max_price is not empty
        properties = properties.filter(
            bath__lte=float(max_bath)
        )  # Filter properties with price less than or equal to max_price
    if min_bath:  # Check if max_price is not empty
        properties = properties.filter(
            bath__gte=float(min_bath)
        )  # Filter properties with price less than or equal to max_price
    if max_sqft:  # Check if max_price is not empty
        properties = properties.filter(
            sq_ft__lte=float(max_sqft)
        )  # Filter properties with price less than or equal to max_price
    if min_sqft:  # Check if max_price is not empty
        properties = properties.filter(
            sq_ft__gte=float(min_sqft)
        )  # Filter properties with price less than or equal to max_price
    regions_df = pd.DataFrame(list(Region.objects.all().values()))

    # properties = Property.objects.all().values()
    properties_df = pd.DataFrame(list(properties.values()))
    df = pd.merge(properties_df,
                  regions_df,
                  left_on='region_id',
                  right_on='id',
                  suffixes=('', '_region'))

    # Split the data
    new_data = df.drop([
        'price', 'address', 'walk_score', 'transit_score', 'bike_score',
        'long', 'lat', 'price_per_sqft', 'name', 'coordinates'
    ],
        axis=1)  # Assuming 'price' is your target variable
    actual_prices = df['price']
    addresses = df['address']
    # loaded_model = tf.keras.models.load_model("property_price_model.h5")
    # loaded_scaler = joblib.load("scaler.pkl")
    loaded_model = tf.keras.models.load_model(
        "/bestproperties/bpaa/property_price_model.h5")
    loaded_scaler = joblib.load("/bestproperties/bpaa/scaler.pkl")
    # Assuming new_data is a new DataFrame with property features
    scaled_new_data = loaded_scaler.transform(new_data)
    predictions = loaded_model.predict(scaled_new_data)
    median_price = df['price'].median()
    mape = mean_absolute_percentage_error(actual_prices,
                                          predictions) / median_price
    print(
        f"Mean Absolute Percentage Error (MAPE) between actual and projected prices: {mape:.2f}"
    )
    # print(predictions)
    differences = [(predicted - actual) / actual
                   for actual, predicted in zip(actual_prices, predictions)]

    # Get indices of top 10 differences
    if not num_of_results:
        num_of_results = 10
    top_n_indices = sorted(range(len(differences)),
                           key=lambda i: differences[i],
                           reverse=True)[:int(num_of_results)]

    # Print the addresses for the top 10 differences
    result = []
    for idx in top_n_indices:
        actual = actual_prices[idx]
        predicted = predictions[idx]
        difference = differences[idx]

        # Extract scalar value if the variables are numpy arrays
        if isinstance(actual, numpy.ndarray) and actual.shape == (1, ):
            actual = actual[0]
        if isinstance(predicted, numpy.ndarray) and predicted.shape == (1, ):
            predicted = predicted[0]
        if isinstance(difference, numpy.ndarray) and difference.shape == (1, ):
            difference = difference[0]
        address_url = addresses[idx].replace(
            ', Austin, TX', ' Austin, TX').replace(' ', '-') + '_rb'

        result.append({
            'address': f'{addresses[idx]}',
            'price': f'${actual:,}',
            'pred': f'${round(predicted):,}',
            'difference': f'{round(difference*100, 2)}%',
            'link': f'https://www.zillow.com/homes/{address_url}'
        })
        print(
            f"Address: {addresses[idx]}, Actual Price: ${actual}, Predicted Price: ${predicted}, Difference: {difference*100}%"
        )
    return result
