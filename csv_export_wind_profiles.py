import pandas as pd
from sklearn.neighbors import NearestNeighbors
from read_data.dowa import read_netcdf
from pytz import utc


def get_dataframe(loc, year=None):
    assert len(loc) == 4
    if loc == 'mmij':
        iy, ix = 111, 56
        ds = read_netcdf(iy - 1, ix - 1, return_ds=True)
    elif loc == 'mmca':
        iy, ix = 74, 99
        ds = read_netcdf(iy - 1, ix - 1, return_ds=True)

    heights = [10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.]
    df_full = ds.to_dataframe().reset_index()

    for i, h in enumerate(heights):
        dfh = df_full.loc[df_full.height == h, ['time', 'wspeed']].set_index('time').rename(columns={'wspeed': 'vw'})
        if i == 0:
            df = dfh
        else:
            df = df.join(dfh, rsuffix='{0:03.0f}'.format(h))
    df = df.rename(columns={'vw': 'vw{0:03.0f}'.format(heights[0])}).reset_index()
    if year is not None:
        df = df[(df.time >= str(year)) & (df.time < str(year+1))]
    df.insert(0, 'id', df.time.apply(lambda t: loc + str(int(t.timestamp()))))
    return df


if __name__ == '__main__':
    loc = 'mmca'
    y = 2008
    df = get_dataframe(loc, year=y)
    print(df.shape[0])
    df.to_csv('wind_profiles_{}_{}.csv'.format(loc, y), index=False)
