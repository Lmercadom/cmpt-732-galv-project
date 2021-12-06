import nltk
from nltk.corpus import wordnet as wn

# synonyms = ['staff']
#
# for syn in wn.synsets("good"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
# print(set(synonyms))

tree = wn.synset('price.n.01')

print(tree.part_meronyms())
print('\n')
print(tree.substance_meronyms())


print(wn.synset('food.n.01').part_holonyms())
print('\n')
print(wn.synset('price.n.01').substance_holonyms())

print(wn.synset('service.v.01').entailments())
