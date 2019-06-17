import pandas as pd

# df_tbl = pd.read_csv('healthData/MIMIC/2019/OUT_LIMIT_NUM_EVENTS_WINSIZE_24H.csv')

## """ Get table header from cvs file """
filename = 'healthData/MIMIC/2019/20_OUT_NUM_EVENTS_WINSIZE_24H.csv'
cols = pd.read_csv(filename, index_col=0, nrows=0).columns

df_tbl = pd.read_csv(filename, usecols=cols.tolist())

cols = df_tbl.columns
cols_ind = [df_tbl.columns.get_loc(c) for c in cols if c in df_tbl]
print(zip(cols, cols_ind))

# print('Table header : ', cols)

#df_tbl2 = pd.read_csv(filename)

#print('Num Cols tbl: ',len(df_tbl.columns))

# Yields a tuple of index label and series for each row in the datafra,e
# for (index_label, row_series) in df_tbl.iterrows():
#     phrase = df_tbl.iloc[index_label,28]
#     if isinstance(phrase, str):
#         from string import maketrans 
#         chars_to_remove = " /``[]"
#         replace_by = '-_    '
#         trantab = maketrans(chars_to_remove, replace_by)
#         phrase =  phrase.translate(trantab)

#         df_tbl.iloc[index_label,28] = phrase

#         print('Row Index label : ', index_label)
#         print('Row Content as Series : ', row_series.values)
#         print(phrase)
#         print(df_tbl.iloc[index_label,28])

# df_tbl.to_csv('healthData/MIMIC/2019/res.csv')

# cols = df_tbl.columns
# cols_ind = [df_tbl.columns.get_loc(c) for c in cols if c in df_tbl]
# print(zip(cols, cols_ind))

# # dateparse = lambda x: pd.to_datetime(str(x), format='%Y-%m-%d %H:%M:%S', errors='coerce')


[('DOB', 0), ('DOB_DAY', 1), ('DOB_HOUR', 2), ('DOB_MIN', 3), ('DOB_MON', 4), ('DOB_SEC', 5), 
('DOB_YEAR', 6), ('DOD', 7), ('DOD_DAY', 8), ('DOD_HOSP', 9), ('DOD_HOUR', 10), ('DOD_MIN', 11), 
('DOD_MON', 12), ('DOD_SEC', 13), ('DOD_SSN', 14), ('DOD_YEAR', 15), ('EXPIRE_FLAG', 16), 
('GENDER', 17), ('HADM_ADMISSION_LOCATION', 18), ('HADM_ADMISSION_TYPE', 19), ('HADM_ADMITTIME', 20), 
('HADM_ADM_DAY', 21), ('HADM_ADM_HOUR', 22), ('HADM_ADM_MIN', 23), ('HADM_ADM_MON', 24), 
('HADM_ADM_SEC', 25), ('HADM_ADM_YEAR', 26), ('HADM_DEATHDAY', 27), ('HADM_DEATHHOUR', 28), 
('HADM_DEATHMIN', 29), ('HADM_DEATHMON', 30), ('HADM_DEATHSEC', 31), ('HADM_DEATHTIME', 32), 
('HADM_DEATHYEAR', 33), ('HADM_DISCHARGE_LOCATION', 34), ('HADM_DISCHTIME', 35), ('HADM_DISCH_DAY', 36), 
('HADM_DISCH_HOUR', 37), ('HADM_DISCH_MIN', 38), ('HADM_DISCH_MON', 39), ('HADM_DISCH_SEC', 40), 
('HADM_DISCH_YEAR', 41), ('HADM_EDOUTTIME', 42), ('HADM_EDREGTIME', 43), ('HADM_ETHNICITY', 44), 
('HADM_HADM_ID', 45), ('HADM_HAS_CHARTEVENTS_DATA', 46), ('HADM_HOSPITAL_EXPIRE_FLAG', 47), 
('HADM_INSURANCE', 48), ('HADM_LANGUAGE', 49), ('HADM_MARITAL_STATUS', 50), ('HADM_RELIGION', 51), 
('HADM_ROW_ID', 52), ('HADM_SUBJECT_ID', 53), ('ICU_DBSOURCE', 54), ('ICU_FIRST_CAREUNIT', 55), 
('ICU_FIRST_WARDID', 56), ('ICU_HADM_ID', 57), ('ICU_ICUSTAY_ID', 58), ('ICU_INTIME', 59), 
('ICU_INTTIME_DAY', 60), ('ICU_INTTIME_HOUR', 61), ('ICU_INTTIME_MIN', 62), ('ICU_INTTIME_MON', 63), 
('ICU_INTTIME_SEC', 64), ('ICU_INTTIME_YEAR', 65), ('ICU_LAST_CAREUNIT', 66), ('ICU_LAST_WARDID', 67),
('ICU_LOS', 68), ('ICU_OUTTIME', 69), ('ICU_OUTTIME_DAY', 70), ('ICU_OUTTIME_HOUR', 71), 
('ICU_OUTTIME_MIN', 72), ('ICU_OUTTIME_MON', 73), ('ICU_OUTTIME_SEC', 74), ('ICU_OUTTIME_YEAR', 75), 
('ICU_ROW_ID', 76), ('ICU_SUBJECT_ID', 77), ('ROW_ID', 78), ('SUBJECT_ID', 79), ('hadm_id', 80), 
('icustay_id', 81), ('procedure', 82), ('subject_id', 83), ('unit', 84)]

