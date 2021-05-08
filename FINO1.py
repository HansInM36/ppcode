import sys
sys.path.append('/scratch/ppcode/')
import funcs

# FINO1 position
lon = funcs.hms2std(6,35,15.58)
lat = funcs.hms2std(54,0,53.94)
x, y = funcs.lonlat2cts(lon, lat)
print(x,y)

# av11 position
lon = funcs.hms2std(6,36,26.970)
lat = funcs.hms2std(54,0,1.218)
x11, y11 = funcs.lonlat2cts(lon, lat)

# av12 position
lon = funcs.hms2std(6,37,11.412)
lat = funcs.hms2std(54,0,1.23)
x12, y12 = funcs.lonlat2cts(lon, lat)


# coordinates of Alpha Ventus
lons = [(6,35,36.592), (6,36,22.836), (6,37,6.030), \
        (6,37,37.834), (6,36,24.220), (6,37,7.866), \
        (6,35,38.970), (6,36,25.569), (6,37,9.546), \
        (6,35,40.158), (6,36,26.970), (6,37,11.412)]

lats = [(54,1,17.976), (54,1,18.066), (54,1,18.050), \
        (54,0,51.569), (54,0,51.581), (54,0,51.570), \
        (54,0,27.018), (54,0,27.012), (54,0,26.994), \
        (54,0,1.212),  (54,0,1.218),  (54,0,1.230)]

xs = []
ys = []

for i in range(12):
    lon, lat = funcs.hms2std(*lons[i]), funcs.hms2std(*lats[i])
    x, y = funcs.lonlat2cts(lon, lat)
    xs.append(x)
    ys.append(y)
    

# origin of palm
x0, y0 = -58288.98149, 6010316.15741

# output PALM coordinates for Alpha Ventus
for i in range(12):
    print(xs[i] - x0, ys[i] - y0)

