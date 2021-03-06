## Running the code
- Clone repository
- Install dependencies
```
pip3 install -r requirements.txt
```
- Download yelp dataset from here [Yelp Dataset](https://www.yelp.com/dataset/download)
- Download Tableau Reader from here [Tableau Reader](https://www.tableau.com/products/reader)

## **ETL Steps**

This step will clean, extract and transform the relevant fields from the yelp-business json for the purpose of visualization in Tableau and use it to feed data to "similar business search CLI" App . The below code writes the results in the `output` folder.
```
spark-submit business-etl.py {downloaded-yelp-dataset}/yelp_academic_dataset_business.json businesses,categories,restaurants,attributes,hours

spark-submit diet-restrictions-etl.py ./output/attributes.parquet output

spark-submit tableau_diet_restriction.py ./output/DietaryRestrictions.json ./output/restaurants.parquet output/diet

spark-submit tableau_attributes.py ./output/attributes.parquet ./output/restaurants.parquet output/restaurant-facilities

spark-submit user_data_ingestion.py {downloaded-yelp-dataset}/yelp_academic_dataset_user.json ./output/user

spark-submit business_data_filter.py {downloaded-yelp-dataset}/yelp_academic_dataset_business.json ./output/restaurants.parquet ./output/data

spark-submit create_reviews_parquet.py ./output/restaurants.parquet {downloaded-yelp-dataset}/yelp_academic_dataset_reviews.json output

# CSV files stored in output/diet and output/restaurant-facilities directories are further used in the Tableau Workbook
# CSV files stored in output/data is further used to feed data for similar business search CLI App.
```

## **ML Code**

```
spark-submit ml.py ./output/restaurants.parquet,./output/attributes.parquet output
```
[Optional] - below command is optional as it is already a part of previous statement. But can be used to test the model.
```
spark-submit test-model.py ./output/model ./output/test-set
```

## **Similar Business Search CLI App** 

`path` : path for filtered business data 

`location_str` : where the business owner wants to open the restaurant

`attributes` : the categories/cuisines user is looking for in the location provided 

`threshold` : the distance in kms that user wants the app to look for businesses from the location provided

```
cd app
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

python3 main.py --path "data" --location_str "vancouver" --attributes "pizza" --threshold "15.0"
cd ..
```

## **Reviews Sentiment Mining**

businesses_id.txt here represents the yelp business ids to perform review mining on.

`**Warning**`: the package `langdetect` used in this script fails to get imported in the cluster. When importing, It tries to find it in the global python packages, but it is installed in the user's python package. It works fine on local machine.

spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4 get_reviews_ngram_counts.py `<reviews file>` `<business ids>` `<ngram>` `<output>`

inputs:

`<reviews file>` : parquet file of reviews

`<business ids>` : txt file with list of businesses ids to perform review mining on. Should be in local not hdfs

`<ngram>` : length of ngram

`<output>` : output folder

`outputs`: n_grams_count.csv

```
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4 get_reviews_ngram_counts.py ./output/reviews.parquet businesses_id.txt 2 output/
```
