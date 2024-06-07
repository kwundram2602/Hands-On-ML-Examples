import numpy as np
from housing_fetch import *

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#housing = hf.load_housing_data()

train_set, test_set = shuffle_and_split_data(housing, 0.2)
#print(len(train_set))
#print(len(test_set))
#
# create data partition 
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
# define id with long an lat
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")


# import test_split 
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set["total_bedrooms"].isnull().sum())

# create an income category attribute with five categories
housing ["income_cat" ] = pd.cut (housing ["median_income" ],
                                  bins =[0., 1.5 , 3.0 , 4.5 , 6., np.inf ], labels =[1, 2, 3, 4, 5])
# show income categories numbers
housing ["income_cat" ].value_counts ().sort_index ().plot .bar (rot=0, grid =True )
plt.xlabel ("Income category" )
plt.ylabel ("Number of districts" )
#plt.show ()

# following code creates 10 different  stratified splits of the same dataset
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit (n_splits =10, test_size =0.2 , random_state =42)
strat_splits = []
for train_index , test_index in splitter .split (housing , housing ["income_cat" ]):
    strat_train_set_n = housing .iloc [train_index ]
    strat_test_set_n = housing .iloc [test_index ]
    strat_splits .append ([strat_train_set_n , strat_test_set_n ])
# using the first split :
strat_train_set , strat_test_set = strat_splits [0]

# other way to get single stratified split with 'stratify' argument:
strat_train_set , strat_test_set = train_test_split ( housing , test_size =0.2 ,
                                                     stratify =housing ["income_cat" ], random_state =42)
# check result 
# print(strat_test_set ["income_cat" ].value_counts () / len(strat_test_set ))
# drop income cat --> reverting data back to its original state
for set_ in (strat_train_set , strat_test_set ):
    set_ .drop ("income_cat" , axis =1, inplace =True )

# copying full training set so one can revert to it afterwards
# if the training set is very large, you may want to sample an exploration set,
# to make manipulations easy and fast during the exploration phase
housing = strat_train_set .copy ()

# show plots if main script
if __name__=="__main__":
    
    #create a scatterplot of all the districts to visualize the data
    housing.plot (kind ="scatter" , x="longitude" , y="latitude" , grid =True )
    #plt.show ()
    
    # Setting thealpha option to 0.2 : makes it  easier to visualize the places where there is a high density of data point
    housing.plot (kind ="scatter" , x="longitude" , y="latitude" , grid =True , alpha =0.2)
    
    # predefined color map from blue to red (jet)
    housing .plot (kind ="scatter" , x="longitude" , y="latitude" ,
                grid =True , s=housing ["population" ] / 100, label ="population" ,
                c="median_house_value" , cmap ="jet" , colorbar =True , legend =True ,
                sharex =False , figsize =(10, 7))
    plt.show ()