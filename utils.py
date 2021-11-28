import boto3
import os


def upload_files(path, bucketName):
    session = boto3.Session(
        profile_name='dev',
        region_name='us-west-2',
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucketName)
    print("Uploading to S3")
    for subdir, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=full_path[len(path)+1:], Body=data)
    print("Uploading to S3 Complete!\n")
