import pandas as pd
import numpy as np

def clean_features(df):
    '''Cleans the feature datasets of nans and irrelevant data.
    Input: pandas dataframe
    Output: pandas dataframe
    '''

    df.drop(['reanalysis_avg_temp_k'], axis=1, inplace=1)
    # Essentially the same information as reanalysis_air_temp_k (pearson
    # correlation of .901).  Interestingly enough, precipitation_amt_mm and
    # reanalysis_precip_amt_kg_per_m2 only have a correlation of .48.  This
    # doesn't make a lot of sense to me, as precip per m2 should simply be the
    # total precipitation divided by the area of the country, which I don't
    # think has changed in the past 5 years.  Same conundrum occurs with
    # reanalysis_relative_humidity_percent and
    # reanalysis_specific_humidity_g_per_kg.

    df.drop(['reanalysis_sat_precip_amt_mm'], axis=1, inplace=1)
    # Same information as precipitation_amt_mm (correlation of 1).

    df.drop(['station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm'], axis=1, inplace=True)
    # This information is all pretty much already in other features with fewer nans.

    df.drop(['week_start_date'], axis=1, inplace=True)
    # df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    # Convert to datetime for easy use in timeseries analysis

    df.drop(['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw'], axis=1, inplace=True)
    # There are a lot of nans, and I don't think these columns will be very important

    df.interpolate(inplace=True)
    # Given the data is a time series and the features are weather, linear interpolation should work fairly well for filling the small number of remaining nans.

    df['sj'] = pd.get_dummies(df['city'], drop_first=True)
    df.drop(['city'], axis=1, inplace=True)
    # Dummy city variable so it is easily handled by certain models.

    return df


if __name__ == '__main__':
    features_train_df = pd.read_csv('../../data/raw/dengue_features_train.csv')
    labels_train_df = pd.read_csv('../../data/raw/dengue_labels_train.csv')
    features_test_df = pd.read_csv('../../data/raw/dengue_features_test.csv')

    clean_features_train = clean_features(features_train_df)
    clean_features_test = clean_features(features_test_df)

    labels_train_df['sj'] = pd.get_dummies(labels_train_df['city'], drop_first=True)
    labels_train_df.drop(['city'], axis=1, inplace=True)

    full_train_df = clean_features_train.merge(labels_train_df, on=['year', 'weekofyear', 'sj'])

    clean_features_test.to_csv('../../data/processed/test_features_clean.csv', index=False)
    full_train_df.to_csv('../../data/processed/train_clean.csv', index=False)
