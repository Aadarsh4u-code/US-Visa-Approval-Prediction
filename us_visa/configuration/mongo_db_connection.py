import os
import sys
from dotenv import load_dotenv
import pymongo
import certifi

from us_visa.constants import COLLECTION_NAME, DATABASE_NAME
from us_visa.exception import CustomException
from us_visa.logger import logging

# Load Constant variables from .env file
load_dotenv()

"""
To prevent connection time-out issues:
Certifi provides Mozilla's carefully curated collection of Root Certificates for validating the trustworthiness of SSL certificates.
"""
ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe 
    
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name = DATABASE_NAME) -> None:
        try:
            logging.info("MongoDB connecting...")
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv('MONGODB_URL_KEY')
                if mongo_db_url is None:
                    raise CustomException(f"Environment key: 'MONGODB_URL_KEY' is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull.")
        except Exception as e:
            raise CustomException(e, sys) from e

# Usage example         
# connection = MongoDBClient(COLLECTION_NAME)
