# Similar Business Search CLI App

An app that let's you look up similar businesses based on "categories" column in business.json data file. Once similar businesses are found it quickly plots them on a map for the user to view it.

![When you search for Pizza in Vancouver](sample_output.png "Demo output")

## Usage
```bash
python main.py --path "data" --location_str "vancouver" --attributes "pizza" --threshold "15.0"
```

## Helps you find your next business opportunity
When a user is undecided on what restaurant/cuisine to open for this app can help you decide that. Just look up "Sushi" in Vancouver around 15kms and it will show all businesses that have 'sushi' ora category or find similar business category if 'sushi' is not available 

## Smart business look up
What if given category is not available in data? 
Fret not, the app uses [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings to first map user provided attribute to an embedding and then it uses [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity#Definition) to find most similar businesses in case the user provided attribute/category is not available in categories column.