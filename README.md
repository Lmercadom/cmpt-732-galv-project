# cmpt-732-galv-project

- Clone repository
- Install dependencies

```
pip3 install -r requirements.txt
```

- Download yelp dataset from here [Yelp Dataset](https://www.yelp.com/dataset/download)

**ETL Steps**

This step will clean, extract and transform the relevant fields from the yelp-business json for the purpose of visualization in Tableau. The below code writes the results in the `output` folder
```
spark-submit business-etl.py {downloaded-yelp-dataset}/yelp-business.json.gz businesses,categories,restaurants,attributes,hours
spark-submit diet-restrictions-etl.py ./output/attributes.parquet output
spark-submit tableau_diet_restriction.py ./output/DietaryRestrictions.json ./output/restaurants.parquet output/diet
spark-submit tableau_attributes.py ./output/attributes.parquet ./output/restaurants.parquet output/restaurant-facilities
```

**ML**

With the below code we want the model to learn how a restaurant facilities affect the rating of a restaurant. We feed 15 attributes/features to the model and predict the rating against the ratings provided by Yelp. In this we find out the features like `WheelchairAccessible` , `DogsAllowed` leads to higher chances in getting a good rating.
```
spark-submit ml.py ./output/attributes.parquet output
```

[Optional] as the above command already prints the test results. Once the training is done you can test the saved model on the test set using the below command. 
```
spark-submit test-model.py ./output/model ./output/test-set
```
