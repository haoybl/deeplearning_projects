 # -*- coding:utf-8 -*-
__author__ = "Wang Hewen"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
import datetime
from collections import defaultdict, OrderedDict
import CommonModules.DataStructureOperations as DSO
import CommonModules.IO as IO
from CommonModules.Utilities import TimeElapsed

def convert_datetime_column_to_integer(column):
    '''
    Calculate how many days between each date and the minimum date.
    '''
    column = (column - column.min()).astype('timedelta64[D]').astype(int)
    return column

def converter_for_float(data):
    if isinstance(data, float) or isinstance(data, int):
        return data
    else:
        return 0


def data_processing():
    print("Load data", TimeElapsed())
    df_train = pd.read_csv('./train_ver2.csv', nrows = 10000, na_values = ["NA", " NA", '     NA', ''])#Detect dtypes using first several rows
    #print(df_train.dtypes)
    df_train = pd.read_csv('./train_ver2.csv', nrows = None, na_values = ["NA", " NA", '     NA'], 
                           dtype = dict(df_train.dtypes), error_bad_lines = False, warn_bad_lines = True, 
                           engine = "c", converters = {"conyuemp": converter_for_float, "indrel_1mes": converter_for_float }, 
                           #usecols = df_train.columns[:12]
                           )#Set nrows to None to load all rows
    print(sum(df_train['conyuemp']))
    del df_train["nomprov"] #Redundant with cod_prov
    df_train["ncodpers"] = df_train["ncodpers"].astype(int)#Convert customer code to integer to prevent potential hash problems.
    date_time_column = ["fecha_dato", "fecha_alta", "ult_fec_cli_1t"]
    for column in date_time_column:
        df_train[column] = pd.to_datetime(df_train[column], format='%Y-%m-%d', errors="coerce")

    print("Add 3 days for date, since date starts at 2015-01-28", TimeElapsed())
    first_day = df_train["fecha_dato"][0]
    df_train["fecha_dato"] = df_train["fecha_dato"] +  datetime.timedelta(days = 3)
    first_day = df_train["fecha_dato"][0]
    first_day_location = (df_train["fecha_dato"] == first_day)
    df_train.loc[first_day_location, "fecha_dato"] = first_day + datetime.timedelta(days = 1)#Move 2015-01-31 to 2015-02-01

    print("Split data month by month", TimeElapsed())
    df_train["behavior_month"] = df_train["fecha_dato"].dt.year * 100 + df_train["fecha_dato"].dt.month
    del df_train["fecha_dato"]

    print("Group by month and customer id", TimeElapsed())
    df_train = df_train.groupby(["behavior_month", "ncodpers"], as_index = False)#as_index is to keep group by labels, behavior_month and ncodpers
    #first_position_function = lambda x: x.iloc[0]
    last_position_function = lambda x: x.iloc[-1]
    aggregate_functions = {}
    aggregate_functions.update(dict.fromkeys(["ind_empleado", "pais_residencia", "sexo", "age", 
                                              "fecha_alta", "ind_nuevo", "antiguedad", "indrel", 
                                              "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", 
                                              "indresi", "indext", "conyuemp", "canal_entrada", 
                                              "indfall", "tipodom", "cod_prov", "ind_actividad_cliente", 
                                              "renta", "segmento"], "last"))#Keep the last occurance of the attribute in the entire month #Not very sure if it's buggy...            
       
    predicted_keys = ["ind_ahor_fin_ult1","ind_aval_fin_ult1","ind_cco_fin_ult1",
                        "ind_cder_fin_ult1","ind_cno_fin_ult1", "ind_ctju_fin_ult1",
                        "ind_ctma_fin_ult1","ind_ctop_fin_ult1", "ind_ctpp_fin_ult1",
                        "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",                                             
                        "ind_ecue_fin_ult1","ind_fond_fin_ult1", "ind_hip_fin_ult1",
                        "ind_plan_fin_ult1","ind_pres_fin_ult1", "ind_reca_fin_ult1",
                        "ind_tjcr_fin_ult1","ind_valo_fin_ult1", "ind_viv_fin_ult1",
                        "ind_nomina_ult1","ind_nom_pens_ult1","ind_recibo_ult1"]
    aggregate_functions.update(dict.fromkeys(predicted_keys, np.max))#When encountering 1 in the entire month, set it as 1

    print("Aggregate by month and customer id", TimeElapsed())
    df_train = df_train.aggregate(aggregate_functions)
    behavior_months = sorted(list(set(df_train["behavior_month"])))#Sort all months from small to big

    #print(df_train["behavior_month"])

    print("Convert categorical data into dummies", TimeElapsed())
    dummy_variables = ["ind_empleado", "pais_residencia", "sexo", "ind_nuevo", 
                       "indrel", "indrel_1mes", "tiprel_1mes", "indresi", "indext", 
                       "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov",
                       "ind_actividad_cliente", "segmento"]
    df_train = pd.get_dummies(df_train, columns = dummy_variables)
    #df_train.drop(dummy_variables, axis = 1, inplace = True) #No need to do this, since the above command has already done
    #df_train = pd.concat([df_train, df_train_dummies], axis = 1)

    #Impute all NAs(Including NaT)
    df_train.fillna(0, inplace = True)

    print("Convert fecha_alta(date) into integer", TimeElapsed())
    df_train["fecha_alta"] = convert_datetime_column_to_integer(df_train["fecha_alta"])
    df_train["ult_fec_cli_1t"] = convert_datetime_column_to_integer(df_train["ult_fec_cli_1t"])
    #Convert related coulumns into integer(No need if read_csv process NA correctly)
    #for column_key in ["age"]:
    #    df_train[column_key] = df_train[column_key].astype(int)

    IO.ExportToPkl("./df_train_temp.pkl", df_train)

    print("Generate monthly data for each user(Perhaps there are better data structure to improve processing efficiency?)", TimeElapsed())
    #Structure:
    #{customer_code1: OrderedDict1([month1: Series1, month2: Series2, month3: [], month4: Series4, ...])}
    #This will be converted into:(Note there are tow Series2)
    #np.array([Series1, Series2, Series2, Series4, ....])
    ncodpers = df_train["ncodpers"].unique()
    unique_months = df_train["behavior_month"].unique()
    Indices = [np.repeat(ncodpers, len(unique_months)), np.tile(unique_months, len(ncodpers))]
    dataset = pd.DataFrame(index = Indices)

    for index, row in df_train.iterrows():
        dataset.loc[row["ncodpers"], row["behavior_month"]] = row.drop(["ncodpers", "behavior_month"], axis = 0)#Since row is a Series, it has only one row
    
    dataset = dict(dataset)
    IO.ExportToPkl("./dataset_temp.pkl", dataset)
    
    print("Generate feature vectors", TimeElapsed())
    feature_vectors_x = np.array([[]])
    feature_vectors_y = np.array([[]])
    for dataset_month_value in dataset.values():
        values_for_single_ncodeper = list(dataset_month_value.values())
        #print(values_for_single_ncodeper)
        if isinstance(values_for_single_ncodeper[0], list):
            for value in values_for_single_ncodeper:
                if not isinstance(value, list):
                    sentinel_value = value
                    break
        else:
            sentinel_value = values_for_single_ncodeper[0]

        for index, value in enumerate(values_for_single_ncodeper):#The row in each behavior_month
            if isinstance(value, list):
                values_for_single_ncodeper[index] = sentinel_value
            else:
                sentinel_value = value

        #print(np.array(list(dataset_month_value.values())))
        feature_vector = pd.DataFrame(list(dataset_month_value.values()))
        feature_vector_y = feature_vector[predicted_keys]
        feature_vector_x = feature_vector.drop(predicted_keys, axis = 1)
        feature_vectors_x = DSO.CombineMatricesRowWise(feature_vectors_x, feature_vector_x, Sparse = False)
        feature_vectors_y = DSO.CombineMatricesRowWise(feature_vectors_y, feature_vector_y, Sparse = False)

    #print(feature_vectors_x.toarray(), feature_vectors_y.toarray())    
    print("Pickling", TimeElapsed())   
    IO.ExportToPkl("./X.pkl", feature_vectors_x)
    IO.ExportToPkl("./y.pkl", feature_vectors_y)
        #print(value)
    return

    #Split into X and y
    y = df_train[predicted_keys]
    #X = df_train[?]

def main():
    pd.options.display.max_columns = None#Display all coulumns
    data_processing()
    #print(df_train.dtypes)
    #print(df_train.head())
    return

if __name__ == "__main__":
    main()