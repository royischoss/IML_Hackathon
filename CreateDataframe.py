import pandas as pd
import numpy as np
from _datetime import datetime

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def create_4_df_splits_processed():
    """
    Creates and returns 4 data splits (original_full, train, validation, test)
    after preprocessing
    """
    original_full = preprocess_data("Dataset/dataset_crimes.csv")
    train = preprocess_data("Dataset/train.csv")
    validation = preprocess_data("Dataset/validate.csv")
    test = preprocess_data("Dataset/test.csv")
    return original_full, train, validation, test


def create_4_df_splits_raw():
    """
    Creates and returns 4 data splits (original_full, train, validation, test)
    using raw data *without* preprocessing
    """
    original_full = pd.read_csv("Dataset/dataset_crimes.csv", index_col=0)
    train = pd.read_csv("Dataset/train.csv", index_col=0)
    validation = pd.read_csv("Dataset/validate.csv", index_col=0)
    test = pd.read_csv("Dataset/test.csv", index_col=0)
    return original_full, train, validation, test


def preprocess_data(path):
    """
    Creates and returns a pandas dataframe from given file path, after
    preprocessing
    """
    df = pd.read_csv(path, index_col=0)

    add_primary_code_col(df)
    date_process(df)
    df = dummies(df)
    drop(df)
    c_names = ['ID', 'Case Number', 'Date', 'Block', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area',
               'X Coordinate', 'Y Coordinate', 'Year', 'Latitude', 'Longitude', 'Location', 'Primary Code',
               'Location Description_ABANDONED BUILDING', 'Location Description_AIRCRAFT',
               'Location Description_AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA',
               'Location Description_AIRPORT BUILDING NON-TERMINAL - SECURE AREA',
               'Location Description_AIRPORT EXTERIOR - NON-SECURE AREA',
               'Location Description_AIRPORT EXTERIOR - SECURE AREA', 'Location Description_AIRPORT PARKING LOT',
               'Location Description_AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA',
               'Location Description_AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
               'Location Description_AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA',
               'Location Description_AIRPORT TERMINAL UPPER LEVEL - SECURE AREA',
               'Location Description_AIRPORT TRANSPORTATION SYSTEM (ATS)',
               'Location Description_AIRPORT VENDING ESTABLISHMENT', 'Location Description_ALLEY',
               'Location Description_ANIMAL HOSPITAL', 'Location Description_APARTMENT',
               'Location Description_APPLIANCE STORE', 'Location Description_ATHLETIC CLUB',
               'Location Description_ATM (AUTOMATIC TELLER MACHINE)',
               'Location Description_AUTO / BOAT / RV DEALERSHIP', 'Location Description_BANK',
               'Location Description_BAR OR TAVERN', 'Location Description_BARBERSHOP',
               'Location Description_BOAT / WATERCRAFT', 'Location Description_BOWLING ALLEY',
               'Location Description_BRIDGE', 'Location Description_CAR WASH', 'Location Description_CEMETARY',
               'Location Description_CHA APARTMENT', 'Location Description_CHA HALLWAY / STAIRWELL / ELEVATOR',
               'Location Description_CHA PARKING LOT / GROUNDS',
               'Location Description_CHURCH / SYNAGOGUE / PLACE OF WORSHIP', 'Location Description_CLEANING STORE',
               'Location Description_COIN OPERATED MACHINE', 'Location Description_COLLEGE / UNIVERSITY - GROUNDS',
               'Location Description_COLLEGE / UNIVERSITY - RESIDENCE HALL',
               'Location Description_COMMERCIAL / BUSINESS OFFICE', 'Location Description_CONSTRUCTION SITE',
               'Location Description_CONVENIENCE STORE', 'Location Description_CREDIT UNION',
               'Location Description_CTA BUS', 'Location Description_CTA BUS STOP',
               'Location Description_CTA PARKING LOT / GARAGE / OTHER PROPERTY', 'Location Description_CTA PLATFORM',
               'Location Description_CTA STATION', 'Location Description_CTA TRACKS - RIGHT OF WAY',
               'Location Description_CTA TRAIN', 'Location Description_CURRENCY EXCHANGE',
               'Location Description_DAY CARE CENTER', 'Location Description_DEPARTMENT STORE',
               'Location Description_DRIVEWAY - RESIDENTIAL', 'Location Description_DRUG STORE',
               'Location Description_FACTORY / MANUFACTURING BUILDING', 'Location Description_FEDERAL BUILDING',
               'Location Description_FIRE STATION', 'Location Description_FOREST PRESERVE',
               'Location Description_GAS STATION', 'Location Description_GOVERNMENT BUILDING / PROPERTY',
               'Location Description_GROCERY FOOD STORE', 'Location Description_HIGHWAY / EXPRESSWAY',
               'Location Description_HOSPITAL BUILDING / GROUNDS', 'Location Description_HOTEL / MOTEL',
               'Location Description_JAIL / LOCK-UP FACILITY',
               'Location Description_LAKEFRONT / WATERFRONT / RIVERBANK', 'Location Description_LIBRARY',
               'Location Description_MEDICAL / DENTAL OFFICE', 'Location Description_MOVIE HOUSE / THEATER',
               'Location Description_NURSING / RETIREMENT HOME', 'Location Description_OTHER (SPECIFY)',
               'Location Description_OTHER COMMERCIAL TRANSPORTATION',
               'Location Description_OTHER RAILROAD PROPERTY / TRAIN DEPOT', 'Location Description_PARK PROPERTY',
               'Location Description_PARKING LOT / GARAGE (NON RESIDENTIAL)', 'Location Description_PAWN SHOP',
               'Location Description_POLICE FACILITY / VEHICLE PARKING LOT', 'Location Description_POOL ROOM',
               'Location Description_RESIDENCE', 'Location Description_RESIDENCE - GARAGE',
               'Location Description_RESIDENCE - PORCH / HALLWAY',
               'Location Description_RESIDENCE - YARD (FRONT / BACK)', 'Location Description_RESTAURANT',
               'Location Description_SCHOOL - PRIVATE BUILDING', 'Location Description_SCHOOL - PRIVATE GROUNDS',
               'Location Description_SCHOOL - PUBLIC BUILDING', 'Location Description_SCHOOL - PUBLIC GROUNDS',
               'Location Description_SIDEWALK', 'Location Description_SMALL RETAIL STORE',
               'Location Description_SPORTS ARENA / STADIUM', 'Location Description_STREET',
               'Location Description_TAVERN / LIQUOR STORE', 'Location Description_TAXICAB',
               'Location Description_VACANT LOT / LAND', 'Location Description_VEHICLE - COMMERCIAL',
               'Location Description_VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS',
               'Location Description_VEHICLE - DELIVERY TRUCK',
               'Location Description_VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)',
               'Location Description_VEHICLE NON-COMMERCIAL', 'Location Description_WAREHOUSE', 'Hour Code_0',
               'Hour Code_1', 'Hour Code_2', 'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
               'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday', 'other']
    d = {}
    for c in c_names:
        d[c] = [0]
    t = pd.DataFrame(d)
    df = df.reindex(columns=t.columns, fill_value=0)
    return df


def add_primary_code_col(df):
    """
    Adds a primary code column to given df.
    """
    conditions = [
        (df["Primary Type"] == "BATTERY"),
        (df["Primary Type"] == "THEFT"),
        (df["Primary Type"] == "CRIMINAL DAMAGE"),
        (df["Primary Type"] == "DECEPTIVE PRACTICE"),
        (df["Primary Type"] == "ASSAULT")
    ]
    values = crimes_dict.keys()
    df["Primary Code"] = np.select(conditions, values)


def date_process(df):
    """
    Processes dates into relative variables.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['hour'] = df['Date'].dt.hour
    conditions = [
        ((df["hour"] > 6) & (df["hour"] <= 14)),
        ((df["hour"] > 14) & (df["hour"] <= 22)),
        ((df["hour"] > 22) | (df["hour"] <= 6))
    ]
    values = [0,1,2]
    df["Hour Code"] = np.select(conditions, values)
    df['day_of_week'] = df['Date'].dt.day_name()
    df['Date'] = ((df['Date'] - datetime(2021, 1, 1)).dt.total_seconds()) / (24 * 60 * 60)


def dummies(df):
    """
    Creates dummy variables.
    """
    mapping = {'TRUE': 1, 'FALSE': 0}
    df = pd.get_dummies(df, columns=['Location Description'])
    df = pd.get_dummies(df, columns=['Hour Code'])
    df = pd.get_dummies(df, columns=['day_of_week'])
    df.replace({'TRUE': mapping, 'FALSE': mapping})
    df['Arrest'] = df['Arrest'].astype(np.int32)
    df['Domestic'] = df['Domestic'].astype(np.int32)
    return df


def drop(df):
    """
    Drops unnecessary columns.
    """
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop('Primary Type', axis=1, inplace=True)
    df.drop('hour', axis=1, inplace=True)

def drop_task_1(df):
    """
    Drops unnecessary columns.
    """
    df.drop('IUCR', axis=1, inplace=True)
    df.drop('Description', axis=1, inplace=True)
    df.drop('FBI Code', axis=1, inplace=True)
    df.drop('Updated On', axis=1, inplace=True)

def split_features_and_labels(df):
    """
    Splits the data to features and labels.
    """
    labels = df["Primary Code"]
    features = df.drop(["Primary Code"], axis=1, inplace=False)
    return features, labels


# ready made datasets, access via import
original_full_p, train_p, validation_p, test_p = create_4_df_splits_processed()
original_full_r, train_r, validation_r, test_r = create_4_df_splits_raw()

train_p_features, train_p_labels = split_features_and_labels(train_p)
validation_p_features, validation_p_labels = split_features_and_labels(validation_p)
test_p_features, test_p_labels = split_features_and_labels(test_p)
