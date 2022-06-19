import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def best_seller(metadata):
    product_group = metadata.groupby('product_id')
    quantity_ordered = product_group.sum()['quantity_ordered']

    products = [product for product, df in product_group]

    plt.bar(products, quantity_ordered)
    plt.ylabel('quantidade vendida')
    plt.xlabel('produto')
    plt.xticks(products, rotation='vertical', size=8)
    plt.show()


def content_based(index):
    index = random.randint(1,3)
    print_recommendation = True

    data = pd.read_csv('products_metadata.csv')

    vectorizer=CountVectorizer()
    series_mat=vectorizer.fit_transform(data['description'])

    cos_sim_data = pd.DataFrame(cosine_similarity(series_mat))
    
    index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:6]
    products_recomm = data['description'].loc[index_recomm].values
    result = {'Produto': products_recomm, 'Index': index_recomm}

    if print_recommendation == True:
        print('O produto comprado é esse: %s \n' % (data['description'].loc[index]))
        k = 1
        for movie in products_recomm:
            print('O produto %i recomendado é o: %s \n' % (k, movie))

    return result

if __name__ == "__main__":
    metadata = pd.read_csv('products_metadata.csv')

    content_based(metadata)
