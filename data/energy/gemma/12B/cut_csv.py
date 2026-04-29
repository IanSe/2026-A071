# cut the csv file from 1779 to the end
import pandas as pd

df = pd.read_csv('./gemma-lora-power_timeseries.csv')
df = df[:1779]
df.to_csv('./gemma-lora-power_timeseries_cut.csv', index=False)