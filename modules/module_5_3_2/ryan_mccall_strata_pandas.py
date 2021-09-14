import pandas as pd
from pandas_datareader import data as pd_data
import numpy as np
import matplotlib.pyplot as plt


p = print
google = pd.read_csv('data/goog.csv', index_col='Date', parse_dates=True)


def data_intro():
    p(pd.__version__)

    p(google)

    start = pd.Timestamp('2010-1-1')
    end = pd.Timestamp('2014-12-31')

    p(type(google))
    p(google.head())
    p(google.info())
    p(google['Open'])
    p(google['Open'].iloc[0: 5])

    p(google.loc['2010-01-04':'2010-01-08', 'Open'])

    google_up = google[google['Close'] > google['Open']]
    p(google_up.head())

    google_filtered = google[pd.isnull(google['Volume']) == False]
    print(google_filtered.head())
    google.describe()


def data_computations():
    google['Return'] = google['Close'].pct_change()
    p(google['Return'].iloc[0:5])
    google['LogReturn'] = np.log(1 + google['Return'])
    p(google['LogReturn'].iloc[0: 5])

    window_size = 252
    google['Volatility'] = google['LogReturn'].rolling(window_size).std() * np.sqrt(window_size)
    p(google['Volatility'].iloc[window_size - 5: window_size + 5])

    p(google.info())
    google[['Close', 'Volatility']].plot(subplots=True, figsize=(15, 6))


def data_structures():
    file = 'data/exoplanets.csv'
    data = pd.read_csv(file)
    series = data['NAME']
    print(f'\n{series}\n{type(series)}')

    new_list = [5, 10, 15, 20, 25]
    new_series = pd.Series(new_list)

    new_dict = {'b': 100, 'a': 200.0, 'd':450, 'c':700}
    print(pd.Series(new_dict))

    from collections import OrderedDict
    od = OrderedDict([('b', 100), ('a', 200), ('d', 450), ('c', 700)])
    pd.Series(od)

    array1 = np.arange(1, 6) * 10.0
    series1 = pd.Series(array1)
    p("\n")
    p(array1)
    p(type(array1))

    print("\n")
    print(series1)
    print(type(series1))
    print(series1.index)
    print(series1.iloc[0])

    index2 = ['a', 'b', 'c', 'd', 'a']
    series2 = pd.Series(np.arange(1, 6) * 10.0, index=index2)

    p(series2.loc['b'])
    p(series2.loc['a'])

    for k, v in series2.iteritems():
        print(k, v)

    print(series1.iloc[1:3])

    p(series2.iloc[1:2])
    p('\nRange of labels (inclusive!) ')
    p(series2.loc['b': 'c'])

    p(series2.iloc[-1:])
    p(series2.iloc[: -3])
    p(series2.loc['d': 'b': -2])

    p(series1 * 2)
    p(series1 * 3)

    series_2pi = pd.Series(np.linspace(-np.pi, np.pi, 100))
    p(series_2pi)

    series_sin = np.sin(series_2pi)
    print(series_sin)
    print(type(series_sin))

    plt.plot(series_2pi, series_sin)
    plt.show()


def operations():
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 20, 30, 40, 50])
    print(series1 + series2)

    series1 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    series2 = pd.Series([10, 20, 30, 40, 50], index=['c', 'd', 'e', 'f', 'g'])
    ssum = series1 + series2
    print(ssum)
    print(ssum.mean())

    print( "Pandas mean: ", pd.Series( [1.0, 2.0, np.nan] ).mean() )
    print( "Numpy mean:  ", np.array(  [1.0, 2.0, np.nan] ).mean() )


def dataframe():
    file = 'data/exoplanets.csv'
    df_file = pd.read_csv(file)
    print(df_file)

    dictionary = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'b': np.linspace(0, np.pi, 10),
                  'c': 0.0,
                  'd': ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]}

    df_dict = pd.DataFrame(dictionary)
    print(df_dict)
    print(df_dict.columns)
    print(df_dict.index)
    p(df_dict.info())

    df_grades = pd.DataFrame({'Alice': [1, 2, 3, 4, 5],
                              'Bob': np.random.random(5),
                              'Chuck': np.random.random(5)},
                             index=['Jan', 'Feb', 'Mar', 'Apr', 'May'])
    series_a = df_grades['Alice']
    print(f'{series_a}\n{type(series_a)}')
    print(df_grades['Alice'].iloc[1])
    print(df_grades.loc['Jan', 'Alice'])

    for k, v in df_grades['Alice'].iteritems():
        print(f'{k} > {v}')
    for i, r in df_grades.iterrows():
        print(f'index {i} row {r["Alice"]}')

    # extract column as new df
    df_col = df_grades[['Alice']]
    # Extract a row
    df_row = df_grades.loc['Jan': 'Jan']

    # slicing along rows
    row_slice = df_grades.loc['Jan': 'Mar', 'Alice']

    # slicing along cols
    col_slice = df_grades.loc[:, 'Alice': 'Bob']

    # sub df
    row_and_col = df_grades.loc['Mar': 'Apr', 'Bob': 'Chuck']

    another_way = df_grades.loc[df_grades['Bob'] < .5]

    my_list = list(df_grades['Alice'])
    print(my_list, type(my_list))
    my_array = np.array(df_grades['Alice'])
    print(my_array, type(my_array))


def io():
    import os
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    file_name = "data/exoplanets.csv"
    data = pd.read_csv(file_name)
    print(type(data))
    print(data.iloc[0:2], data.index)

    # use column to create a better index using date
    data = pd.read_csv(file_name, parse_dates=True, index_col='DATE')
    p(data.head(10))
    p(data.index)
    data = data.set_index(data.index.sort_values(ascending=False))
    print(data.iloc[0: 3])

    data['price'] = 1e6
    print(data.price.iloc[0: 5])
    del data['FIRSTURL']
    print(data.iloc[0:5])
    print(data.head())
    print(data.columns)

    # question 1
    print('ave PERIOD %s' % data['PERIOD(day)'].mean())

    # question 2
    plt.plot(data['PERIOD(day)'])
    # plt.show()  # there is a spike in 2014

    # question 3
    data['price'] = 1e6 * data['MASS(mjupiter)'] - 25000 * (data['DIST(PARSEC)'] - 10)
    print(data['price'])

# Define a convenience function to help us clean up
def clean_tmp(file_name="tmp/exoplanet.csv"):
    if os.path.isfile(file_name):
        os.remove(file_name)


def file_formats():
    # all read_* methods in pands
    print("".join(["pd.%s\n" % reader
                   for reader in dir(pd)
                   if reader.startswith('read_')]))

    # DataFrame can be cast to files or other objects
    print("".join(["pd.DataFrame.%s\n" % reader
                   for reader in dir(pd.DataFrame)
                   if reader.startswith('to_')]))

    df = pd.read_csv('data/exoplanets.csv',
                     parse_dates=['DATE'],
                     encoding='utf-8')

    df = pd.read_csv('data/beer2.csv.gz', index_col=0, parse_dates=['time'])
    print(df.head())
    print(df.beer_style)
    print(df.beer_style.str)  # StringMethods object
    mask = df.beer_style.str.contains('[A|a]merican')
    print(df.beer_style[mask])

    print(df.time.dt.hour)

    p(df['beer_name'][0: 3])

    reviews = df.set_index(['profile_name', 'beer_id', 'time']).sort_index()
    p(reviews.head(5))

    # decrease memory usage by using categoricals
    df['beer_style'] = df['beer_style'].astype('category')
    p(df[['beer_style']].info())
    df.drop


def group_by():
    pass


if __name__ == '__main__':
    group_by()













