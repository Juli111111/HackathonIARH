import pandas as pd
import uuid


df = pd.read_csv("HRDataset_v14-1.csv")



df["Employee_Name"] = ["EMP_NUM" + str(i).zfill(4) for i in range(len(df))]
df["EmpID"] = [str(i).zfill(4) for i in range(len(df))]
df["ManagerName"] = "ANON_MANAGER"



if "Zip" in df.columns:
    
    df = df.drop(columns=["Zip"])



ethnic_columns = ["RaceDesc", "HispanicLatino"]

for col in ethnic_columns:
    if col in df.columns:
        df[col] = "removed"


df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce").dt.year







df.to_csv("employees_anonymized.csv", index=False)
