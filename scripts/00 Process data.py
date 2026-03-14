import os
import polars as pl
from pyprojroot import here

output_path = str(here()) + "/data/processed/"
os.mkdir(output_path) if not os.path.exists(output_path) else None


def load_file_name(folder_path):
    file_name_list = []
    for file in os.listdir(folder_path):
        if file.endswith("txt") and "_" in file:
            file_name_list.append(file)
    return file_name_list


def load_one_df(
    file_name,
    col_ls=(
        ["UNIT_NUM"]
        + ["CYCLE"]
        + ["OS" + "_" + str(i) for i in range(1, 4)]
        + ["SM" + "_" + str(i) for i in range(1, 22)]
    ),
):
    df = pl.read_csv(
        str(here()) + "/data/raw/" + file_name, separator=" ", has_header=False
    )
    df = df.select(
        pl.all().exclude([col for col in df.columns if df[col].null_count() == len(df)])
    )

    if len(df.columns) >= len(col_ls):
        df.columns = col_ls
        df = df.with_columns(pl.lit(file_name).alias("FILE_NAME"))
    else:
        df.columns = ["RUL"]
        df = df.with_row_index("UNIT_NUM", 1)

    return df


ls = load_file_name("data/raw")
df_ls = [load_one_df(i) for i in ls]

RUL_1 = df_ls[0]
RUL_2 = df_ls[1]
RUL_3 = df_ls[2]
RUL_4 = df_ls[3]

RUL_1 = RUL_1.rename({"RUL": "MAX_RUL"})
RUL_2 = RUL_2.rename({"RUL": "MAX_RUL"})
RUL_3 = RUL_3.rename({"RUL": "MAX_RUL"})
RUL_4 = RUL_4.rename({"RUL": "MAX_RUL"})

test_1 = df_ls[4]
test_2 = df_ls[5]
test_3 = df_ls[6]
test_4 = df_ls[7]

train_1 = df_ls[8]
train_2 = df_ls[9]
train_3 = df_ls[10]
train_4 = df_ls[11]

train_1 = train_1.with_columns(pl.col("CYCLE").max().over("UNIT_NUM").alias("MAX_RUL"))
train_2 = train_2.with_columns(pl.col("CYCLE").max().over("UNIT_NUM").alias("MAX_RUL"))
train_3 = train_3.with_columns(pl.col("CYCLE").max().over("UNIT_NUM").alias("MAX_RUL"))
train_4 = train_4.with_columns(pl.col("CYCLE").max().over("UNIT_NUM").alias("MAX_RUL"))

test_1 = test_1.join(RUL_1, on="UNIT_NUM", how="left")
test_2 = test_2.join(RUL_2, on="UNIT_NUM", how="left")
test_3 = test_3.join(RUL_3, on="UNIT_NUM", how="left")
test_4 = test_4.join(RUL_4, on="UNIT_NUM", how="left")

train_1 = train_1.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
train_2 = train_2.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
train_3 = train_3.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
train_4 = train_4.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))

test_1 = test_1.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
test_2 = test_2.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
test_3 = test_3.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))
test_4 = test_4.with_columns((pl.col("MAX_RUL") - pl.col("CYCLE")).alias("RUL"))

train = pl.concat([train_1, train_2, train_3, train_4])
test = pl.concat([test_1, test_2, test_3, test_4])

train.write_csv(str(here()) + "/data/processed/train.csv")
test.write_csv(str(here()) + "/data/processed/test.csv")
