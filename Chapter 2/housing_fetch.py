from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
def load_housing_data ():
    tarball_path = Path ("datasets/housing.tgz" )
    if not tarball_path.is_file ():
        Path ("datasets" ).mkdir (parents =True , exist_ok =True )
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib .request.urlretrieve (url , tarball_path )
        with tarfile .open (tarball_path ) as housing_tarball :
            housing_tarball .extractall (path ="datasets" )
    return pd.read_csv (Path ("datasets/housing/housing.csv" ))
housing = load_housing_data ()


print("Fetch imported")
if __name__=="__main__":
    #housing.head()
    #housing.info ()
    #print(housing["ocean_proximity" ].value_counts())
    print(housing.describe())
    housing.hist (bins =50, figsize =(12, 8))
    plt.show ()