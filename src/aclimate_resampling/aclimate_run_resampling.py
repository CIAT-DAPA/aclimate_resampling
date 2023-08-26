import datetime
import pandas as pd
import os
import argparse

from dateutil.relativedelta import relativedelta

from resampling import Resampling
from complete_data import CompleteData

def main():
    # Params
    # 0: Country
    # 1: Path root
    # 2: Previous months
    # 3: Cores
    # 4: Year of forecast

    parser = argparse.ArgumentParser(description="Resampling script")

    parser.add_argument("-C", "--country", help="Country name", required=True)
    parser.add_argument("-p", "--path", help="Path to data directory", default=os.getcwd())
    parser.add_argument("-m", "--prev-months", type=int, help="Previous months", required=True)
    parser.add_argument("-c", "--cores", type=int, help="Number of cores", required=True)
    parser.add_argument("-y", "--forecast-year", type=int, help="Forecast year", required=True)

    args = parser.parse_args()

    print("Reading inputs")
    print(args)

    #country = "ETHIOPIA"
    country = args.country
    #path = "D:\\CIAT\\Code\\USAID\\aclimate_resampling\\data\\"
    path = args.path
    #start_date = (datetime.date.today() - pd.DateOffset(months=1)).replace(day=1)
    months_previous = args.prev_months
    start_date = (datetime.date.today() - pd.DateOffset(months=months_previous)).replace(day=1)
    cores = args.cores
    
    ar = Resampling(path, country, year_forecast = args.forecast_year)
    ar.resampling()
    dd = CompleteData(start_date,country,path,cores=cores)
    dd.run()

if __name__ == "__main__":
    main()

#python resampling.py "ETHIOPIA" "D:\\CIAT\\Code\\USAID\\aclimate_resampling\\data\\" "-1" 2 2023