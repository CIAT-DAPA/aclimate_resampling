   # -*- coding: utf-8 -*-
  # Functions to do climate daily data forecast per station
  # Created by: Maria Victoria Diaz
  # Alliance Bioversity, CIAT. 2023

import pandas as pd
import numpy as np
import os
import warnings
import dask.dataframe as dd
from datetime import datetime


warnings.filterwarnings("ignore")

class Resampling():

  def __init__(self,path,country, year_forecast, current_month):
     self.path = path
     self.country = country
     self.path_inputs = os.path.join(self.path,self.country,"inputs")
     self.path_inputs_prediccion = os.path.join(self.path_inputs,"prediccionClimatica")
     self.path_inputs_daily = os.path.join(self.path_inputs_prediccion,"dailyData")
     self.path_outputs = os.path.join(self.path,self.country,"outputs")
     self.path_outputs_pred = os.path.join(self.path_outputs,"prediccionClimatica")
     self.path_outputs_res = os.path.join(self.path_outputs_pred,"resampling")
     self.path_outputs_prob = os.path.join(self.path_outputs_pred,"probForecast")

     self.year_forecast = year_forecast
     self.current_month = current_month
     self.npartitions = 10 #int(round(cores/3)) 

     pass

  def mdl_verification(self,daily_weather_data, prob_root):


      clima = os.listdir(daily_weather_data)
      clima = [file for file in clima if not file.endswith("_coords.csv")]
      clima = [file.split(".csv")[0] for file in clima]

      prob = pd.read_csv(os.path.join(prob_root , "probabilities.csv"))



      prob = prob[prob['id'].isin(clima)]



      check_clm = []
      for i in range(len(clima)):
          df = pd.read_csv(os.path.join(daily_weather_data, f"{clima[i]}.csv"))

          # 1. max de temp_max == min de temp_max
          # 2. max de temp_min == min de temp_min
          # 3. max de srad == min de srad

          max_tmax = df['t_max'].max()
          min_tmax = df['t_max'].min()

          max_tmin = df['t_min'].max()
          min_tmin = df['t_min'].min()

          max_srad = df['sol_rad'].max()
          min_srad = df['sol_rad'].min()

          if max_tmax == min_tmax or max_tmin == min_tmin or max_srad == min_srad:
              resultado = pd.DataFrame({'code': [clima[i]], 'value': [f"tmax = {max_tmax}; tmin = {max_tmin}; srad = {max_srad}"]})
          else:
              resultado = pd.DataFrame({'code': [clima[i]], 'value': ["OK"]})
          check_clm.append(resultado)

      df = pd.concat(check_clm)
      df_1 = df[df['value'] == "OK"]
      df_2 = df[df['value'] != "OK"]

      code_ok = df_1
      code_problema = df_2


      # 1. Probabilidades con cero categoria normal
      # 2. Probabilidades sumen > 1.1
      # 3. Probabilidades sumen <  0.9

      prob['sum'] = prob['below'] + prob['normal'] + prob['above']
      prob.loc[prob['normal'] == 0.00, 'normal'] = -1
      prob.loc[prob['sum'] < 0.9, 'normal'] = -1
      prob.loc[prob['sum'] > 1.1, 'normal'] = -1

      df_1 = prob[prob['normal'] == -1]
      df_2 = prob[(prob['normal'] >= 0) & (prob['normal'] <= 1)]

      code_p_ok = df_2
      code_p_malos = df_1

      ids_buenos = code_p_ok['id'].tolist()

      result_clima_prob_outside = list(set(clima) - set(code_p_ok['id']))

      code_problema = pd.DataFrame({'ids': [1] + code_problema['code'].tolist(),
                                    'descripcion': [None] + code_problema['value'].tolist()})
      code_p_malos = pd.DataFrame({'ids': [1] + code_p_malos['id'].tolist(),
                                  'descripcion': "Problemas base de datos probabilidad"})

      result_clima_prob_outside = pd.DataFrame({'ids': [1] + result_clima_prob_outside,
                                                'descripcion': "La estacion esta fuera de area predictora"})

      ids_malos = pd.concat([code_problema, code_p_malos, result_clima_prob_outside])
      ids_buenos = pd.DataFrame({'ids': ids_buenos})
      ids_malos = ids_malos.replace(1, pd.NA).dropna()

      result = {'ids_buenos': ids_buenos, 'ids_malos': ids_malos}
      return result

  def preprocessing(self,prob_root,  ids):

    """ Determine seasons of analysis according to the month of forecast in CPT

    Args:

    prob_root: str
              The root of the probabilities file from CPT, with its name and extension.


    ids: dict
              Dictionary with a list of stations with problems and not to be analyzed, and
              a list of stations without problems.

    Returns:

      Dataframe
          a dataframe with the original columns + name of season + start month of season +
          end month of season

    """


    # Read the CPT probabilities file

    proba = pd.read_csv(os.path.join(prob_root , "probabilities.csv"))

    ids_x = ids['ids_buenos']
    prob = proba[proba['id'].isin(ids_x['ids'])]
    s = prob['season'].iloc[0]
    forecast_period = "tri" if s.count("-") > 1 else "bi"

    # Check the period of forecast
    if forecast_period == "tri":

      # Create a list of month numbers from 1 to 12, followed by [1, 2] to create quarters
      months_numbers =list(range(1,13)) + [1,2]

      # Create a DataFrame representing periods of three consecutive months (with its numbers)
      period= pd.DataFrame( [months_numbers[i:i+3] for i in range(0, len(months_numbers)-2)])
      period.columns = ['Start', 'Central_month', 'End']


      # Merge the prob DataFrame with the period DataFrame based on the 'month' and 'Central_month' columns
      prob = prob.merge(period, left_on='month', right_on='Central_month')
      prob.drop(['month','Central_month'], axis = 1, inplace = True )

    else:
      if forecast_period == "bi":

        # Create a list of month numbers from 1 to 12
        months_numbers = list(range(1,13))
        months_numbers.append(1)


        # Create a DataFrame representing periods of two consecutive months (with its numbers)
        period = pd.DataFrame( [months_numbers[i:i+2] for i in range(0, len(months_numbers)-1)])
        period.columns = ['Start', 'End']


        # Merge the prob DataFrame with the period DataFrame based on the 'month' and 'Start' month columns
        prob = prob.merge(period, left_on='month', right_on='Start')

        # Merge the prob DataFrame with the period DataFrame based on the 'month' and 'End' month columns
        # Join with prob_a
        #prob = prob_a.append(prob.merge(period, left_on='month', right_on='End'))
        prob.drop(['month'], axis = 1, inplace = True )

    # Reshape the 'prob' DataFrame and put the 'below', 'normal' and 'above' probability categories in a column
    prob = prob.melt(id_vars = ['year', 'id', 'predictand','season', 'Start','End'], var_name = 'Type', value_name = 'Prob')

    #Return probability DataFrame
    return prob, forecast_period
  

  def gen_muestras(self, new_data, prob_type):
    
         
    subset = new_data.loc[new_data['condition'] == prob_type]
    m = subset.sample(1)

    if any(m['year'] == max(new_data['year'])):
      m = subset[subset['year'] != max(new_data['year'])].sample(1)
    else:
       m = m
    
    return m['year']
  

  def process_escenario(self, data, season, year,index):

      if season == 'Nov-Dec-Jan':
              m1 = data[(data['month'].isin([11,12])) & (data['year']== year)]
              m2 = pd.concat([m1, data[(data['month'] == 1) & (data['year'] == year+1)]])
              m2['index'] = index
      else:
              if season == 'Dec-Jan-Feb':
                m1 = data[(data['month'] == 12) & (data['year'] == year)]
                m2 = pd.concat([m1,data[(data['month'].isin([1,2])) & (data['year'] == year+1)]])
                m2['index'] = index
                
              else:
                    if season == 'Dec-Jan':
                        m1 = data[(data['month'] == 12) & (data['year'] == year)]
                        m2 = pd.concat([m1,data[(data['month'] == 1) & (data['year'] == year + 1)]])
                        m2.loc['index'] = index
                    else:
                        m2 = data[data['year'] == year]
                        m2['index'] = index
      return m2

  def forecast_station(self, station, prob, daily_data_root, output_root, year_forecast, forecast_period):
    
    """ Generate  forecast scenaries

    Args:

    station: str
            The id of th station

      prob: DataFrame
              The result of preprocessing function

      daily_data_root: str
              Where the climate data by station is located

      output_root: str
              Where outputs are going to be saved.

      year_forecast: int
              Year to forecast

      forecast_period: str
              'bi' if the period of CPT forecast is bimonthly.
              'tri' if the period of CPT forecast is quarter.

    Returns:

      Dataframe
          a dataframe with climate daily data for every season and escenary id
          a dataframe with years of escenary for every season

    """
    # Create folders to save result
    val_root = os.path.join(output_root, "validation")
    if not os.path.exists(val_root):
        os.mkdir(val_root)

    # Read the climate data for the station
    clim = pd.read_csv(os.path.join(daily_data_root ,f"{station}.csv"))

    # Filter the probability data for the station
    cpt_prob = prob[prob['id']==station]

    if len(cpt_prob.index) == 0:
      print('Station does not have probabilites')
      base_years = 0
      seasons_range = 0
      p = {'id': [station],'issue': ['Station does not have probabilites']}
      problem = pd.DataFrame(p)

      return base_years, seasons_range,  problem

    else:
      # Get the season for the forecast
      season = np.unique(cpt_prob['season'])
      tri_seasons = ['Dec-Jan-Feb', 'Jan-Feb-Mar', 'Feb-Mar-Apr', 'Jan-Feb', 'Feb-Mar']

      # Adjust the year if the forecast period is 'tri' if necessary
      if  any(np.isin(season, tri_seasons)) :
         year_forecast = year_forecast+1

      # Check if year of forecast is a leap year for February
      leap_forecast = (year_forecast%400 == 0) or (year_forecast%4==0 and year_forecast%100!=0)

      # Filter the February data for leap years
      clim_feb = clim.loc[clim['month'] == 2]
      clim_feb['leap'] = [True if (year%400 == 0) or (year%4==0 and year%100!=0) else False for year in clim_feb['year']]

      # Standardize february months by year according to year of forecat
      february = pd.DataFrame()
      for i in np.unique(clim_feb['year']):
        year_data =  clim_feb.loc[clim_feb['year']==i,:]
        year = year_data.loc[:,'leap']

        # If year of forecast is a leap year and a year in climate data is not, then add one day to february in climate data
        if leap_forecast == True and year.iloc[0] == False:
          year_data = pd.concat([year_data, year_data.sample(1)], ignore_index=True)
          year_data.iloc[-1,0] = 29
        else:

          # If year of forecast is not a leap year and a year in climate data is, then remove one day to february in climate data
          if leap_forecast == False and year.iloc[0] == True:
            year_data =  year_data.iloc[:-1]
          else:

            # If both year of forecast and year in climate data are leap years or not, then keep climate data the same
            year_data = year_data
        february = pd.concat([february, year_data])


      # Concat standardized february data with the rest of climate data
      data = february.drop(['leap'], axis = 1 )
      data = pd.concat([data,clim.loc[clim['month'] != 2]]).sort_values(['year','month'])

      # Start the resampling process for every season of analysis in CPT probabilities file

      base_years =  pd.DataFrame() # List to store years of sample for each season
      seasons_range = pd.DataFrame() # List to store climate data in the years of sample for each season

      for season in  list(np.unique(cpt_prob['season'])):

        # Select the probabilities for the season
        x = cpt_prob[cpt_prob['season'] == season]


        predictand = cpt_prob['predictand'].iloc[0]


      # Compute total precipitation for each year in the climate data range selected
        new_data = data[['year',predictand]].groupby(['year']).sum().reset_index()

        data['season'] = season


      # Calculate quantiles to determine precipitation conditions for every year in climate data selected
        cuantiles = list(np.quantile(new_data['prec'], [.33,.66]))
        new_data['condition'] =  'NA'
        new_data.loc[new_data[predictand]<= cuantiles[0], 'condition'] = 'below'
        new_data.loc[new_data[predictand]>= cuantiles[1], 'condition'] =  'above'
        new_data.loc[(new_data[predictand]> cuantiles[0]) & (new_data[predictand]< cuantiles[1]), 'condition'] =  'normal'

      # Sample 100 records in probability file of season based on probability from CPT as weights
        muestras = x[['Start', 'End', 'Type', 'Prob']].sample(100, replace = True, weights=x['Prob'])
        muestras = muestras.set_index(pd.Index(list(range(0,100))))

        muestras_by_type = []
        for i in range(len(muestras)):
              m = self.gen_muestras(new_data, muestras.iloc[i]['Type'])
              muestras_by_type.append(m)           


        # Join the 100 samples and add sample id
        muestras_by_type = pd.concat(muestras_by_type).reset_index()
        muestras_by_type['index'] = muestras.index

        # Rename year column with season name
        muestras_by_type = muestras_by_type.rename(columns={'year': season})

        # Set the sample years as a list and sort
        years = list(muestras_by_type[season])

        p = pd.DataFrame()
        for x in range(len(years)):
            p1 = self.process_escenario(data=data, season=season, year=years[x], index=muestras_by_type.iloc[x]['index'])
            p = pd.concat([p, p1], ignore_index=True)

        # Join seasons samples by column by sample id
        base_years = pd.concat([base_years, muestras_by_type[['index',season]]], axis = 1,ignore_index=True)

        # Join climate data filtered for the seasons
        seasons_range = pd.concat([seasons_range, p])


      seasons_range = seasons_range.rename(columns = {'index': 'id'})

      if (forecast_period == 'tri') and (len(list(np.unique(cpt_prob['season']))) == 2):

            s = list(np.unique(cpt_prob['season']))
            base_years = base_years.iloc[:,[0,1,3] ]
            base_years = base_years.rename(columns={0: 'id',1: s[0], 3: s[1]})
            base_years['id'] = base_years['id'] + 1
            seasons_range['id'] = seasons_range['id']+1
            seasons_range = seasons_range.sort_values(by=['year', 'month'], ascending=True)
            base_years.to_csv(os.path.join(val_root,  f"{station}_Escenario_A.csv"), index = False)


            #Return climate data filtered with sample id
            return base_years, seasons_range

      else:
          if (forecast_period == 'bi') and (len(list(np.unique(cpt_prob['season']))) == 3) :

            s = list(np.unique(cpt_prob['season']))
            base_years = base_years.iloc[:,[0,1,3,5] ]
            base_years = base_years.rename(columns={0: 'id',1: s[0], 3: s[1], 5: s[2]})
            base_years['id'] = base_years['id'] + 1
            seasons_range['id'] = seasons_range['id']+1
            seasons_range = seasons_range.sort_values(by=['year', 'month'], ascending=True)
            base_years.to_csv(os.path.join(val_root,  f"{station}_Escenario_A.csv"), index = False)


            #Return climate data filtered with sample id
            return base_years, seasons_range

          else:

            print('Station does not have all the seasons availables')

            s = list(np.unique(cpt_prob['season']))
            if len(base_years.columns) == 2:
              base_years = base_years.iloc[:,[0,1] ]
              base_years = base_years.rename(columns={0: 'id',1: s[0]})
            else:
              if len(base_years.columns == 4):
                base_years = base_years.rename(columns={0: 'id',1: s[0], 3: s[1]})
              else:
                base_years = base_years.rename(columns={0: 'id',1: s[0]})

            base_years['id'] = base_years['id'] + 1
            seasons_range['id'] = seasons_range['id']+1


            p = {'id': [station],'issue': ['Station does not have all the seasons availables'], 'Seasons available': ", ".join([str(item) for item in s])}
            problem = pd.DataFrame(p)
            print(problem)
            base_years.to_csv(os.path.join(val_root, f"{station}_Escenario_A.csv"), index = False)

            #Return climate data filtered with sample id
            return base_years, seasons_range, problem


  def add_year(self, year_forecast, observed_month, current_month):
  
    if observed_month < current_month:
     a = year_forecast + 1
    else:
      a = year_forecast

    return a



  def save_forecast(self, station, output_root, year_forecast, seasons_range, base_years, current_month):


    if isinstance(base_years, pd.DataFrame):
    # Set the output root based on forecast period
      output_estacion = os.path.join(output_root, station)
      if not os.path.exists(output_estacion):
          os.mkdir(output_estacion)
          print("Path created for the station: {}".format(station))

      output_summary = os.path.join(output_root, "summary")
      if not os.path.exists(output_summary):
          os.mkdir(output_summary)

      escenarios = []
      IDs= list(np.unique(seasons_range['id']))
      year_forecast = int(year_forecast)
      
      for i in range(len(IDs)):
          df = seasons_range[(seasons_range['id'] == IDs[i])]
          df = df.reset_index()
          df = df.drop(columns = ['year'])

          for j in list(range(len(df))):          
              df.loc[j, 'year'] = self.add_year(year_forecast = year_forecast, observed_month= df.loc[j, 'month'], current_month= current_month)

          df = df.drop(['index','id', 'season'], axis = 1)
          df['year'] = df['year'].astype('int')

          escenarios.append(df)
          df.to_csv(os.path.join(output_estacion ,f"{station}_escenario_{str(i+1)}.csv"), index=False)

      print("Escenaries saved in {}".format(output_estacion))

      # Calculate maximum and minimum of escenaries by date and save
      df = pd.concat(escenarios)
      columns = list(df.columns)
      columns.remove('year')
      new_columns = columns[:2] + ['year'] + columns[2:]
      df = df[new_columns]
      

      df.groupby(['year', 'month', 'day']).max().reset_index().sort_values(['month', 'day'], ascending = True).to_csv(os.path.join(output_summary, f"{station}_escenario_max.csv"), index=False)
      df.groupby(['year', 'month', 'day']).min().reset_index().sort_values(['month', 'day'], ascending = True).to_csv(os.path.join(output_summary, f"{station}_escenario_min.csv"), index=False)
      print("Minimum and Maximum of escenaries saved in {}".format(output_summary))

      vars = df.columns
      vars = [item for item in vars if item != "year"]
      vars = [item for item in vars if item != "month"]
      vars = [item for item in vars if item != "day"]

      accum = df.groupby(['id', 'month'])['prec'].sum().reset_index().rename(columns = {'id': 'escenario_id'})#.sort_values(['id', 'month'], ascending = True).reset_index()#
      prom = df.groupby(['id', 'month'])[vars].mean().rename(columns = {'id': 'escenario_id'})#.reset_index()#.sort_values(['id', 'month'], ascending = True).reset_index()#.rename(columns = {vars[i]: 'max'})

      summary = pd.merge(accum, prom, on=["escenario_id", "month"])

      summary_min = summary.groupby(['month']).min().reset_index().drop(['escenario_id'], axis = 1)#.sort_values(['id', 'month'], ascending = True).reset_index()#.rename(columns = {vars[i]: 'max'})
      summary_min = self.add_year(summary_min, year_forecast, current_month=current_month)

      summary_max = summary.groupby(['month']).max().reset_index().drop(['escenario_id'], axis = 1)
      summary_max = self.add_year(summary_max, year_forecast, current_month=current_month)

      
      summary_avg = summary.groupby(['month']).mean().reset_index().drop(['escenario_id'], axis = 1)
      summary_avg = self.add_year(summary_avg, year_forecast, current_month=current_month)

      vars = [item for item in vars if item != "id"]
      vars.append('prec')

      for i in range(len(vars)):
         

         summary_min[['year','month',vars[i]]].sort_values(['year', 'month'], ascending = True).to_csv(os.path.join(output_summary, f"{station}_{vars[i]}_min.csv"), index=False)
         summary_max[['year','month',vars[i]]].sort_values(['year', 'month'], ascending = True).to_csv(os.path.join(output_summary, f"{station}_{vars[i]}_max.csv"), index=False)
         summary_avg[['year','month',vars[i]]].sort_values(['year', 'month'], ascending = True).to_csv(os.path.join(output_summary, f"{station}_{vars[i]}_avg.csv"), index=False)



      print("Minimum, Maximum and Average of variables by escenary is saved in {}".format(output_summary))
      return df


    else:

      return None
    
    

  def master_processing(self,station, input_root, climate_data_root, verifica ,output_root,  year_forecast, current_month):


    if not os.path.exists(output_root):
        os.mkdir(output_root)       
        print("Path created for outputs")

    print("Reading the probability file and getting the forecast seasons")
    prob_normalized = self.preprocessing(input_root, verifica)


    print("Resampling and creating the forecast scenaries")
    resampling_forecast = self.forecast_station(station = station,
                                           prob = prob_normalized[0],
                                           daily_data_root = climate_data_root,
                                           output_root = output_root,
                                           year_forecast = year_forecast,
                                           forecast_period= prob_normalized[1])


    print("Saving escenaries and a summary")
    self.save_forecast(station = station,
                  output_root = output_root,
                  year_forecast = year_forecast,
                  base_years = resampling_forecast[0],
                  seasons_range = resampling_forecast[1],
                  current_month= current_month)

    if len(resampling_forecast) == 3:
        oth =os.path.join(output_root, "issues.csv")
        resampling_forecast[2].to_csv(oth, mode='a', index=False, header=not os.path.exists(oth))

    else:
        return None
    

  def resampling(self):


    
    print("Fixing issues in the databases")
    verifica = self.mdl_verification(self.path_inputs_daily, self.path_outputs_prob)
    
    
    estaciones = os.listdir(self.path_inputs_daily)
    n = [i for i in estaciones if not  i.endswith("_coords.csv") ]
    n = [i.replace(".csv","") for i in n]
    n1 = [i for i in n if i in list(verifica['ids_buenos']['ids'])]

  
    print("Processing resampling for stations")

    
    n_df = pd.DataFrame(n1,columns=["id"])
    n_df_dd = dd.from_pandas(n_df,npartitions=self.npartitions)
    _col = {'id': object
            ,'issue':object
            , 'season': object
            , 'name': object}
    sample = n_df_dd.map_partitions(lambda df:
                                    df["id"].apply(lambda x: self.master_processing(station = x,
                                               input_root =  self.path_outputs_prob,
                                               climate_data_root = self.path_inputs_daily,
                                               output_root = self.path_outputs_res,
                                               verifica = verifica,
                                               year_forecast = self.year_forecast,
                                               current_month= self.current_month)
                                                  ), meta=_col
                                                  ).compute(scheduler='processes')
    return sample
   
  