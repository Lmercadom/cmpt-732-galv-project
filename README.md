# cmpt-732-galv-project

- Clone repository
- Install dependencies

```
pip3 install -r requirements.txt
```

- Download yelp dataset from here [Yelp Dataset](https://www.yelp.com/dataset/download)
- Download Tableau Reader from here [Tableau Reader](https://www.tableau.com/products/reader)

**ETL Steps**

This step will clean, extract and transform the relevant fields from the yelp-business json for the purpose of visualization in Tableau. The below code writes the results in the `output` folder
```
spark-submit business-etl.py {downloaded-yelp-dataset}/yelp-business.json.gz businesses,categories,restaurants,attributes,hours
spark-submit diet-restrictions-etl.py ./output/attributes.parquet output
spark-submit tableau_diet_restriction.py ./output/DietaryRestrictions.json ./output/restaurants.parquet output/diet
spark-submit tableau_attributes.py ./output/attributes.parquet ./output/restaurants.parquet output/restaurant-facilities
# CSV files stored in output/diet and output/restaurant-facelities directories are further used in the Tableau Workbook
```

**ML**

With the below code we want the model to learn how the facelities of a restaurant affect its ratings. We feed 15 attributes/features to the model and predict the rating against the ratings provided by Yelp. This could be used to predict the initial ratings of a new restaurant based on the facelities it provides. In this model, we find out that features like `WheelchairAccessible` , `DogsAllowed` are given more weightage while predicting the rating. We get an accuracy of ~82%.
```
spark-submit ml.py ./output/attributes.parquet output
```

[Optional] as the above command already prints the test results. Once the training is done you can test the saved model on the test set using the below command. 
```
spark-submit test-model.py ./output/model ./output/test-set
```

**Tableau**

The Tableau Workbook is already connected to the CSV files that were created during the above ETL steps. Dietary Restriction Dashboard gives the number of restaurants that offer food for people with dietary restrictions such as gluten allergy, lactose intolerance etc. and also plots their location on a map. Restaurant Facelities Dashboard shows for each state, the total number of restaurants available for various kinds of services. This can further be drilled down for each city.

<img align=center width="900" alt="Dietary Restrictions" src="https://user-images.githubusercontent.com/24526992/145533791-83ba2f08-2a1c-452d-a14a-20eb288cf2c9.png">

<img align=center width="900" alt="Restaurant Facelities" src="https://user-images.githubusercontent.com/24526992/145534040-a4e847a4-00be-4b20-a4f4-af675a15ff35.png">
