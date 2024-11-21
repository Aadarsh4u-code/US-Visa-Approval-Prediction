from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import CustomException

import pandas as pd
import sys
from typing import Optional
import numpy as np

class UsVisaDataDB:
    """
    This class help to export entire MongoDB record as pandas Dataframe.
    """
    def __init__(self):
        """Create MongoDB Client to access database"""
        try:
            self.mongodb_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]=None) -> pd.DataFrame:
        """
        database_name (str [Optional]): Name of Database inside collection/table is created.
        collection_name (str): Name of the Collection/Table inside data is stored.
        """
        try:
            if database_name is None:
                collection = self.mongodb_client.database[collection_name]
            else:
                collection = self.mongodb_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns="_id", axis=1)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e


