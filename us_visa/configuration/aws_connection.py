import os
import boto3
import sys


from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, REGION_NAME
from dotenv import load_dotenv

# Load Constant variables from .env file
# load_dotenv(dotenv_path='/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/.env')
load_dotenv()

class S3Client:
    s3_resources = None
    s3_client = None
    
    def __init__(self, region_name = os.getenv(REGION_NAME)):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """
        try:
            if S3Client.s3_resources is None or S3Client.s3_client is None:
                __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
                __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
                print('__access_key_id',__access_key_id)
                print('__secret_access_key',__secret_access_key)
                if __access_key_id is None:
                    raise CustomException(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not not set.")
                if __secret_access_key is None:
                    raise CustomException(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
                S3Client.s3_resources = boto3.resource('s3', aws_access_key_id = __access_key_id, aws_secret_access_key = __secret_access_key, region_name = region_name)
                S3Client.s3_client = boto3.resource('s3', aws_access_key_id = __access_key_id, aws_secret_access_key = __secret_access_key, region_name = region_name)
            self.s3_resources = S3Client.s3_resources
            self.s3_client = S3Client.s3_client
        except Exception as e:
            raise CustomException(e, sys) from e


d= S3Client()
