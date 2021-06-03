
import pandas as pd
import gmplot

df = pd.read_csv('Dataset/train.csv')
df.dropna(inplace=True)
gmapf = gmplot.GoogleMapPlotter(41.868397, -87.648281, 13)
gmapf.apikey = "AIzaSyDeRNMnZ__VnQDiATiuz4kPjF_c9r1kWe8"

df_th = df.loc[df["Primary Type"]=="THEFT"]
df_bat = df.loc[df["Primary Type"]=="BATTERY"]
df_ass = df.loc[df["Primary Type"]=="ASSAULT"]

th_lon = df_th["Longitude"]
th_lat = df_th["Latitude"]
gmapf.scatter(th_lat, th_lon, color='red', marker=True)


bat_lon = df_bat["Longitude"]
bat_lat = df_bat["Latitude"]
gmapf.scatter(bat_lat, bat_lon, color='blue', marker=True)

ass_lon = df_ass["Longitude"]
ass_lat = df_ass["Latitude"]
gmapf.scatter(ass_lat, ass_lon, color='green')

gmapf.draw("map.html")
