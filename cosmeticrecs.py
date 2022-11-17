"""DS 2500 FINAL PROJECT

cosmetics price predictor and product recommendation system
devyanshi chandra, kirti magam"""

import pandas as pd
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def clean_df():
    """
    Returns
    -------
    cos_df : dataframe
        cleaned dataframe of sephora cosmetics data.

    """
    # read csv into dataframe
    df = pd.read_csv("sephora_website_dataset.csv")

    # drop unneeded columns
    df = df.drop(["URL", "how_to_use", "MarketingFlags",
                  "MarketingFlags_content", "online_only", "exclusive",
                  "limited_edition", "limited_time_offer", "price",
                  "ingredients", "options", "love"], axis=1)

    # drop rows w/o size values
    df = df[df["size"].str.contains("oz") == True]

    # reformatting descriptions, change to lowercase and remove unusable
    # characters
    df["description"] = df["details"].str.lower()
    df["description"] = df["description"].str.replace(r"[Ã¢â‚¬â„¢:ËœÃ‚#ÃƒÅ“/Â©â€°]",
                                                      " ", regex=True)
    df["description"] = df["description"].str.replace("\n",
                                                      " ", regex=True)
    df["description"] = df["description"].str.replace("  ", " ", regex=True)

    # convert name, cat, and brand strings to lower case to analyze and reset
    # idx
    df["category"] = df["category"].str.lower()
    df["name"] = df["name"].str.lower()
    df["brand"] = df["brand"].str.lower()

    df = df.reset_index()

    # create copy and remove extra characters in size column
    cos_df = df.copy()

    for i in range(len(cos_df)):
        value = cos_df.loc[i, "size"].lower()
        cos_df.at[i, "size"] = extract_size(value)

    return cos_df


def extract_size(value):
    """
    Parameters
    ----------
    value : string
        string value in size column of csv.

    Returns
    -------
    value : float
        cleaned size value converted to float.

    """

    # split numeric value from measurement/oz
    value = value.split("oz")
    value = value[0]

    # continue to split characters from numeric value
    value_lst = re.split("/|;|,| - |\(|packettes", value)

    # if there's a split, take second value as numeric value
    if len(value_lst) > 1:
        value = value_lst[1]

    # replace blank spaces
    value = value.replace(" ", "")

    # check for unneeded strings in value
    if "x" in value:
        # split to separate size and amt of the item
        size = value.split("x")
        amt = float(size[1].replace(" ", ""))
        # split based on other words present in string
        if "pans" in size[0] or "satchets" in size[0]:
            size = size[0].split("pans")
            size = size[0].split("satchets")
        # multiply quantity by amt to get total size in oz
        quant = float(size[0])
        value = quant * amt

    # addressing other if cases w/ their own methods to retrieve float value of
    # size
    elif "fl" in value:
        value = float(value.split("fl")[0])

    elif "pencil:" in value:
        value = float(value.split("pencil:")[1])
    elif "pencil" in value:
        value = float(value.split("pencil")[1])
    elif "liquid" in value:
        value = float(value.split("liquid")[1])
    elif "glaze:" in value:
        value = float(value.split("glaze:")[1])
    elif "four" in value:
        value = float(value.split("four")[1]) * 4
    else:
        value = float(value)

    return value


def tagging_non_numeric(df, brands_dict, category_dict):
    """

    Parameters
    ----------
    df : dataframe
        dataframe of cosmetics data.
    brands_dict : dict
        empty dictionary to keep track of brands .
    category_dict : dict
        empty dictionary to keep trak of product categories.

    Returns
    -------
    df : dataframe
        cosmetics dataframe w/ brand column and category column changes to
        numeric values.

    """

    # initialize counts for brands and categories
    brand_val = 0
    cat_val = 0

    for i in range(len(df)):
        # retrieve brand and category at current row of dataframe
        brand = df.loc[i, "brand"].lower()
        category = df.loc[i, "category"].lower()

        # every time new brand is added to dictionary, it receieves a number
        # the number increments by one for each new company
        if brand not in brands_dict:
            brands_dict[brand] = brand_val
            brand_val += 1
            df.loc[i, "brand val"] = brands_dict[brand]
        # if the brand already exists in the dictionary, the brand is replaced
        # with its numeric tag
        elif brand in brands_dict:
            df.loc[i, "brand val"] = brands_dict[brand]

        # every time new category is added to dictionary, it receieves a number
        # the number increments by one for each new company
        if category not in category_dict:
            category_dict[category] = cat_val
            cat_val += 1
            df.loc[i, "category val"] = category_dict[category]
        # if the category already exists in the dictionary, the category is
        # replaced with its numeric tag
        elif category in category_dict:
            df.loc[i, "category val"] = category_dict[category]

    return df


def price_vals(df):
    """

    Parameters
    ----------
    df : dataframe
        dataframe with cosmetics data.

    Returns
    -------
    df : dataframe
        dataframe w/ price column changed to numeric categories.

    """

    # assigning prices to numbered categories based on what price range they
    # fall into and replacing the price with this value
    for i in range(len(df)):
        price = float(df.loc[i, "value_price"])
        if price > 0 and price <= 15:
            df.loc[i, "price cat"] = 0
        elif price > 15 and price <= 30:
            df.loc[i, "price cat"] = 1
        elif price > 30 and price <= 45:
            df.loc[i, "price cat"] = 2
        elif price > 45 and price <= 60:
            df.loc[i, "price cat"] = 3
        elif price > 60 and price <= 75:
            df.loc[i, "price cat"] = 4
        elif price > 75 and price <= 90:
            df.loc[i, "price cat"] = 5
        elif price > 90 and price <= 105:
            df.loc[i, "price cat"] = 7
        elif price > 105:
            df.loc[i, "price cat"] = 8

    return df


def run_knn_classifier(X_train, y_train, X_test, y_test,
                       K, output):
    """

    Parameters
    ----------
    X_train : list
        list of the features for the training set.
    y_train : list
        list of the targets for the training set.
    X_test : list
        list of the features for the test set.
    y_test : list
        list of the targets for the test set.
    K : int
        k-neighbors value.
    output : boolean
        true or false for how format of report is returned.

    Returns
    -------
    report : dict
        dict of accuracy, precision, and recall metrics.

    """

    y_train = y_train.values.ravel()

    # makes a classifier object
    knn = KNeighborsClassifier(K)
    knn.fit(X=X_train, y=y_train)

    # predicts the targets of the test set
    predicted = knn.predict(X=X_test)
    actual = y_test

    # evaluation of dat: precision and recall
    report = metrics.classification_report(actual, predicted,
                                           output_dict=output)

    return report


def product_recs(df, title):
    """

    Parameters
    ----------
    df : dataframe
        dataframe of cosmetics data.
    title : string
        brand or product name desired for recommendation.

    Returns
    -------
    recommended_products : lst
        returns a list of recommended products based on cnt vectorizors and
        cosine similarity

    """

    # make string lowercase
    title = str(title).lower()

    cat = ""

    # iterate through dataframe to check existence of
    for i in range(len(df)):
        # retrieve brand and product name at current row
        brand = df.loc[i, "brand"].lower()
        name = df.loc[i, "name"].lower()

        # if they entered a brand
        if title == brand and cat == "":
            while True:
                # filter dataframe to only include products of brand they chose
                df2 = df[df["brand"].str.contains(title) == True]
                # find categories for the brand they entered
                cats = df2["category"].unique().tolist()
                # take in product category input from user
                print("Please pick from this list of products available:")
                print()
                print(cats)
                cat = str(input("What type of product would you like?: "))
                print()
                cat = cat.lower()
                # check if the category they chose exists for that brand or
                # reprompt them
                if cat in cats:
                    break
                else:
                    print("The category you put in is incorrect",
                          "please pick something from the list")
                    print()
            # filter dataframe to only includes products of their desired
            # category
            filter_df = df2[df2["category"].str.contains(cat) == True]

            # randomly select a product from that brand and category they chose
            # b/c they didn't know a product name
            name_idx = random.randint(0, (len(filter_df) - 1))
            prod_name = filter_df["name"].values[name_idx]
            print("We'll be giving you products based on this product:",
                  prod_name.title())
        # if they entered a specific product name
        else:
            # find product name in dataframe based on user's input
            if title == name:
                prod_name = title
            else:
                pass

    # referenced :
    # https://thingsgrow.me/2020/04/14/testing-nlp-based-product-recommendation-algorithms-through-sephora/

    # indices used to keep track of the product name
    indices = pd.Series(df.name)

    # initialize the count vectorizer
    count = CountVectorizer()
    # convert descriptions to real number vectors
    count_matrix = count.fit_transform(df["description"])

    # calculate cosine similarities between description to see how close they are
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # create a recommendation list
    recommended_products = []

    # receives the index of the product name
    idx = indices[indices == prod_name].index[0]

    # creates a Series based on cosine similarity of description and sorts
    # retrieve the top ten indices corresponding to most similar products
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1:11].index)

    # adding these products to recommended products list
    for i in top_10_indexes:
        product = df.loc[i, "name"]
        prod_category = df.loc[i, "category"]
        recommended_products.append([product, prod_category])

    return recommended_products


def recommend_viz(df):
    """

    Parameters
    ----------
    df : dataframe
        dataframe of cosmetics data .

    Returns
    -------
    visualization plot of accuracy for each trial

    """
    # randomly choosing 50 product names from product names in dataframe to run
    # trials on recommendation system
    product_names = df["name"].tolist()
    random_products = random.sample(product_names, k=50)
    product_df = df.copy()

    # keeping track of correct percentages and trial counts for each iteration
    correct_percent = []
    trials = []
    trial_count = 1
    for product in random_products:
        # taking name and category for that specific product from dataframe
        df2 = product_df.loc[product_df["name"] == product]
        product_cat = df2["category"].tolist()
        product_cat = product_cat[0]

        # calling product rec function and receiving results
        recs_lst = product_recs(df, product)
        count = 0

        # going through each product in rec list and if its category matches
        # with the category of the product sent into the function, add to the
        # count
        for rec in recs_lst:
            if product_cat == rec[1]:
                count += 1
        # calculate the percentage of correct categories from rec list
        percent = round((count / len(recs_lst)) * 100, 2)
        correct_percent.append(percent)
        trials.append(trial_count)
        trial_count += 1

    # if the percentage of correctly matched categories is greatet than 50,
    # plot the point green, otherwise plot it as red
    greater_count = 0
    less_count = 0
    for i in range(len(correct_percent)):
        if correct_percent[i] > 50:
            plt.scatter(trials[i], correct_percent[i], c="lightgreen")
            greater_count += 1
            if greater_count == 1:
                plt.scatter(trials[i], correct_percent[i], c="lightgreen",
                            label="> 50%")
        else:
            plt.scatter(trials[i], correct_percent[i], c="salmon")
            less_count += 1
            if less_count == 1:
                plt.scatter(trials[i], correct_percent[i], c="salmon",
                            label="<= 50%")

    # adding plot title and labels
    plt.title("Category Accuracy of Product Rec Lists")
    plt.xlabel("Trial Number")
    plt.ylabel("Percentage Correct")
    plt.legend(loc="upper right")
    plt.show()


def price_analysis(X, y):
    """

    Parameters
    ----------
    X : dataframe
        dataframe containing features for classification.
    y : dataframe
        dataframe containing target values .

    Returns
    -------
    returns plots on price prediction analyzing accuracy metrics for all k values
    between 3 and 15
    also returns plot as well as the individual accuracy metrics for each price
    category while k = 4

    """

    # use train test split to randomy split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                        test_size=0.25)

    # keep track of precision and recall per category when k = 4
    k4_precision = []
    k4_rec = []

    # keep track of accuracy, precision, and recall when changing values of k
    # with price prediction
    accuracy = []
    precision = []
    recall = []

    # trying k values ranging from 3 to 14
    for i in range(3, 15):

        # run the knn classifier for each k value
        current_r = run_knn_classifier(X_train, y_train, X_test,
                                       y_test, K=i, output=True)
        # work through report dictionary to retreive accuracy, precision, recall
        # numbers totaled from each category
        total_p = 0
        total_r = 0
        for label, value in current_r.items():
            if type(value) is dict:
                for k, v in value.items():
                    if k == "precision":
                        total_p += v
                        # when k = 4 (the best trial), keep track of precision
                        # for first visualization
                        if i == 4:
                            k4_precision.append(v)
                    elif k == "recall":
                        total_r += v
                        # when k = 4 (the best trial), keep track of accuracy
                        # for first visualization
                        if i == 4:
                            k4_rec.append(v)
            else:
                if label == "accuracy":
                    accuracy.append((100 * value))

        # calculate precision and recall percentages from avg of all categories
        precision.append(100 * (total_p / len(current_r)))
        recall.append(100 * (total_r / len(current_r)))

    k4_rec = k4_rec[:8]
    k4_precision = k4_precision[:8]

    # creating price range labels
    xlabels = ["0 - 15", "15 - 30", "30 - 45", "45 - 60", "60 - 75", "75 - 90",
               "90 - 105", "105+"]
    x = np.arange(len(k4_rec))

    # plotting
    plt.bar(x - 0.2, k4_rec, color="peachpuff", label="Recall", width=0.4)
    plt.bar(x + 0.2, k4_precision, color="peru", label="Precision", width
    =0.4)
    plt.xticks(x, xlabels, rotation=35)
    plt.xlabel("Price Categories in Dollars")
    plt.ylabel("Precision and Recall for each Label")
    plt.title("Precision and Recall Amounts per Price Category (k = 4)")
    plt.legend()
    plt.show()

    plt.plot(range(3, 15), accuracy, color="maroon", label="Accuracy %")
    plt.plot(range(3, 15), precision, color="steelblue", label="Precision %")
    plt.plot(range(3, 15), recall, color="darkmagenta", label="Recall %")
    plt.xlabel("K-values used for K Neighbors")
    plt.ylabel("Percentages for Metrics")
    plt.title("Accuracy, Recall, and Precision %s of Classifier Based on K-Value")
    plt.legend()
    plt.show()


def main():
    # call function to clean data
    df = clean_df()

    # intialize dictionaries for brand and product categories to keep track
    # when they're tagged with a numeric value
    brands_dict = {}
    category_dict = {}

    # call tagging function and price category function
    df = tagging_non_numeric(df, brands_dict, category_dict)
    df = price_vals(df)

    # create feature and target dataframe
    X = df[["brand val", "category val", "size"]]
    y = df[["price cat"]]

    # call price prediction analysis function
    price_analysis(X, y)

    """
    product recc visualization:
        runs recommendation 50 times to plot visualization-- takes a few minutes
        to finish running
    """
    recommend_viz(df)

    # product recommendation function call

    # creating list of unique brands and names in dataframe
    brands = df["brand"].unique().tolist()
    names = df["name"].unique().tolist()

    for brand in brands:
        brand = brand.lower()

    for name in names:
        name = name.lower()

    # asking for product name/brand and checking if it exists & reprompting if
    # it doesn't
    while True:
        title = str(input("Enter the name of a product or brand you like to"
                          " find others like it: ")).lower()
        if title in brands or title in names:
            product_lst = product_recs(df, title)
            break
        else:
            print("Sorry, this does not exist in the database. Please reenter")
            print()

    # printing out product recommendation list to the user
    for i in range(len(product_lst)):
        product = str(i + 1) + ": " + str(product_lst[i][0]).title()
        print(product)


if __name__ == "__main__":
    main()















