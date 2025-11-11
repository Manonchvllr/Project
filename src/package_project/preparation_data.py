
import pandas


def merge_data(data1, data2):
    base =  pandas.merge(data1, data2, on=["DEP", "AAAA", "MM"], how="left").drop_duplicates()
    col = base.columns.tolist()
    col[2], col[14] = col[14], col[2]
    base = base[col]
    base.head
    return(base)


def process_data():pass