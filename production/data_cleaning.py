"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)
from scripts import binned_selling_price


@register_processor("data-cleaning", "social-media-data")
def clean_social_media_data_table(context, params):
    """Clean the ``SOCIAL MEDIA`` data table.

    The table mentions of theme across all Social media Platforms. This
    includes information on theme id, published date of the themes and
    the number of total posts on that day.
    """

    input_dataset = "raw/social_media"
    output_dataset = "cleaned/social_media"

    # load dataset
    social_media_data = load_dataset(context, input_dataset)

    social_media_data_clean = (
        social_media_data.copy()
        # set dtypes : nothing to do here
        .passthrough()
        # Dropping null values from the dataset
        .dropna()
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"published_date": "date", "Theme Id": "Claim_Id"})
        # Convert argument to datetime.
        .to_datetime("date")
    )

    # save the dataset
    save_dataset(context, social_media_data_clean, output_dataset)

    return social_media_data_clean


@register_processor("data-cleaning", "sales-data")
def clean_sales_data_table(context, params):
    """Clean the ``SALES`` data table.

    The table shows the product and date wise sales
    in terms of sales dollars, sales units and sales lbs.
    """

    input_dataset = "raw/sales"
    output_dataset = "cleaned/sales"

    # load dataset
    sales_data = load_dataset(context, input_dataset)

    sales_data_clean = (
        sales_data
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough()
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"system_calendar_key_N": "date"})
    )
    # Convert argument to datetime.
    sales_data_clean["date"] = pd.to_datetime(
        sales_data_clean["date"].astype(str), format="%Y-%m-%d"
    )
    # save the dataset
    save_dataset(context, sales_data_clean, output_dataset)

    return sales_data_clean


@register_processor("data-cleaning", "google-search-data")
def clean_google_search_data_table(context, params):
    """Clean the ``GOOGLE SEARCH`` data table.

    The table the platform with the respective date, year and
    week on which a particular theme was searched about and the count of searches.
    """

    input_dataset = "raw/google"
    output_dataset = "cleaned/google"

    # load dataset
    google_search_data = load_dataset(context, input_dataset)

    google_search_data_clean = (
        google_search_data
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough().drop(["year_new", "week_number"], axis=1)
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"Claim_ID": "Claim_Id"})
        # Convert argument to datetime.
        .to_datetime("date", format="%d-%m-%Y")
    )

    # save the dataset
    save_dataset(context, google_search_data_clean, output_dataset)

    return google_search_data_clean


@register_processor("data-cleaning", "product-manufacturer-list")
def clean_product_manufacturer_list_table(context, params):
    """Clean the ``PRODUCT MANUFACTURE LIST`` data table.

    The table mentions the product with its manufacturer or vendor.
    """

    input_dataset = "raw/product"
    output_dataset = "cleaned/product"

    # load dataset
    product_manufacturer_list = load_dataset(context, input_dataset)

    product_manufacturer_list_clean = (
        product_manufacturer_list[["PRODUCT_ID", "Vendor"]]
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough()
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"PRODUCT_ID": "product_id"})
    )

    # save the dataset
    save_dataset(context, product_manufacturer_list_clean, output_dataset)

    return product_manufacturer_list_clean


@register_processor("data-cleaning", "Theme-Product-list")
def clean_theme_product_list_table(context, params):
    """Clean the ``Theme Product list`` data table.

    The table maps the product with its respective theme.
    """

    input_dataset = "raw/Theme_Product_list"
    output_dataset = "cleaned/Theme_Product_list"

    # load dataset
    Theme_product_list = load_dataset(context, input_dataset)

    theme_product_list_clean = (
        Theme_product_list
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough()
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"PRODUCT_ID": "product_id", "CLAIM_ID": "Claim_Id"})
    )

    # save the dataset
    save_dataset(context, theme_product_list_clean, output_dataset)

    return theme_product_list_clean


@register_processor("data-cleaning", "Theme-list")
def clean_theme_list_table(context, params):
    """Clean the ``Theme list`` data table.

    The table contains the theme names along with its Id .
    """

    input_dataset = "raw/Theme_list"
    output_dataset = "cleaned/Theme_list"

    # load dataset
    Theme_list = load_dataset(context, input_dataset)

    theme_list_clean = (
        Theme_list
        # while iterating on testing, it's good to copy the dataset(or a subset)
        # as the following steps will mutate the input dataframe. The copy should be
        # removed in the production code to avoid introducing perf. bottlenecks.
        .copy()
        # set dtypes : nothing to do here
        .passthrough()
        # .rename(columns = {'test':'TEST'}, inplace = True)
        .rename(columns={"Claim Name": "Claim_name", "CLAIM_ID": "Claim_Id"})
    )
    # save dataset
    save_dataset(context, theme_list_clean, output_dataset)
    return theme_list_clean


@register_processor("data-cleaning", "sales-data-processed")
def process_sales_product_table(context, params):
    """Prepare the ``SALES Processed `` data table.

    The table is a summary table obtained by doing a ``inner`` join of the
    ``SALES``, ```PRODUCT```, ```THEME PRODUCT LIST``` and ``THEME LIST`` tables.
    Then calculating the vendor wise sales and merging the dataframes together for further processing.
    """
    input_sales_ds = "cleaned/sales"
    input_product_ds = "cleaned/product"
    input_theme_product_ds = "cleaned/Theme_Product_list"
    input_theme_list_ds = "cleaned/Theme_list"
    output_dataset = "processed/sales_data_processed"

    # load datasets
    sales_data_clean = load_dataset(context, input_sales_ds)
    product_manufacturer_list_clean = load_dataset(context, input_product_ds)
    theme_product_list_clean = load_dataset(context, input_theme_product_ds)
    theme_list_clean = load_dataset(context, input_theme_list_ds)

    # merging the datasets together
    sales_product_data = sales_data_clean.merge(
        product_manufacturer_list_clean, how="inner", on="product_id"
    )
    sales_product_data = sales_product_data.merge(
        theme_product_list_clean, how="inner", on="product_id"
    )
    sales_product_data = sales_product_data.merge(
        theme_list_clean, how="inner", on="Claim_Id"
    )

    # Dropping the "product_id" column from the sales_product_data
    sales_product_data.drop("product_id", axis=1, inplace=True)

    # Calculating the per unit price for all the records.
    sales_product_data["per_unit_price"] = (
        sales_product_data["sales_dollars_value"]
        / sales_product_data["sales_units_value"]
    )

    # Finding the unit price for every client
    client_A_sales = (
        sales_product_data[sales_product_data["Vendor"] == "A"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg(
            {
                "sales_dollars_value": "sum",
                "sales_units_value": "sum",
                "sales_lbs_value": "sum",
                "per_unit_price": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "sales_dollars_value": "client_A_sales_dollars_value",
                "sales_units_value": "Client_A_sales_units_value",
                "sales_lbs_value": "client_A_sales_lbs_value",
                "per_unit_price": "Client_A_sales_unit_price",
            }
        )
    )
    client_B_sales = (
        sales_product_data[sales_product_data["Vendor"] == "B"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "B_sales_unit_price"})
    )
    client_E_sales = (
        sales_product_data[sales_product_data["Vendor"] == "E"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "E_sales_unit_price"})
    )
    client_D_sales = (
        sales_product_data[sales_product_data["Vendor"] == "D"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "D_sales_unit_price"})
    )
    client_F_sales = (
        sales_product_data[sales_product_data["Vendor"] == "F"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "F_sales_unit_price"})
    )
    client_G_sales = (
        sales_product_data[sales_product_data["Vendor"] == "G"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "G_sales_unit_price"})
    )
    client_H_sales = (
        sales_product_data[sales_product_data["Vendor"] == "H"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "H_sales_unit_price"})
    )
    client_Private_sales = (
        sales_product_data[sales_product_data["Vendor"] == "Private Label"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "Private_sales_unit_price"})
    )
    client_other_sales = (
        sales_product_data[sales_product_data["Vendor"] == "Others"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"per_unit_price": "mean"})
        .reset_index()
        .rename(columns={"per_unit_price": "others_sales_unit_price"})
    )

    # Merging the unit price for all the vendors in our dataframe for comparison
    A_B_sales = client_A_sales.merge(
        client_B_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_sales = A_B_sales.merge(
        client_E_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_D_sales = A_B_E_sales.merge(
        client_D_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_D_F_sales = A_B_E_D_sales.merge(
        client_F_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_D_F_G_sales = A_B_E_D_F_sales.merge(
        client_G_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_D_F_G_H_sales = A_B_E_D_F_G_sales.merge(
        client_H_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    A_B_E_D_F_G_H__Private_sales = A_B_E_D_F_G_H_sales.merge(
        client_Private_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    sales_data_processed = A_B_E_D_F_G_H__Private_sales.merge(
        client_other_sales, on=["date", "Claim_Id", "Claim_name"], how="left"
    )
    sales_data_processed.fillna(0, inplace=True)

    save_dataset(context, sales_data_processed, output_dataset)
    return sales_data_processed


@register_processor("data-cleaning", "social-media-processed")
def process_social_media_data_processed_table(context, params):
    """Prepare the ``SOCIAL MEDIA Processed `` data table.

    The table is a summary table obtained by aggregating the data at weekly level
    and doing a ``inner`` join of the ``SOCIAL MEDIA`` and ``THEME LIST`` tables.
    """
    input_social_media_ds = "cleaned/social_media"
    input_theme_list_ds = "cleaned/Theme_list"

    output_dataset = "processed/social_media_data_processed"

    # load datasets
    social_media_data_clean = load_dataset(context, input_social_media_ds)
    theme_list_clean = load_dataset(context, input_theme_list_ds)

    # merging the datasets together
    social_media_data_processed = social_media_data_clean.merge(
        theme_list_clean, on="Claim_Id"
    )

    # Setting date as index
    social_media_data_processed.set_index("date", inplace=True)

    # Aggregating the data at weekly level till saturday of every week.
    social_media_data_processed = social_media_data_processed.groupby(
        [
            "Claim_Id",
            "Claim_name",
            pd.Grouper(freq="W-SAT", closed="right", label="right"),
        ]
    ).sum()
    social_media_data_processed = social_media_data_processed.reset_index()

    save_dataset(context, social_media_data_processed, output_dataset)
    return social_media_data_processed


@register_processor("data-cleaning", "google-search-data-processed")
def process_social_media_data_processed_table(context, params):
    """Prepare the ``GOOGLE SEACH Processed `` data table.

    The table is a summary table obtained by aggregating the data at weekly level
    and doing a ``inner`` join of the ``GOOGLE SEARCH`` and ``THEME LIST`` tables.
    Then calculating the platform wise searches and using the dataset for further processing.
    """
    input_google_ds = "cleaned/google"
    input_theme_list_ds = "cleaned/Theme_list"

    output_dataset = "processed/google_search_data_processed"

    # load datasets
    google_search_data_clean = load_dataset(context, input_google_ds)
    theme_list_clean = load_dataset(context, input_theme_list_ds)

    # merging the datasets together
    google_search_data2 = google_search_data_clean.merge(
        theme_list_clean, on="Claim_Id"
    )

    # Setting date as index
    google_search_data2.set_index("date", inplace=True)

    # Aggregating the data at weekly level till saturday of every week.
    df_weekly_g = (
        google_search_data2.groupby(["platform", "Claim_Id", "Claim_name"])
        .resample("W-SAT")
        .sum()
    )
    df_weekly_g.drop("Claim_Id", axis=1, inplace=True)
    df_weekly_g = df_weekly_g.reset_index()
    df_weekly_g

    # Calculating the platform wise searches
    amazon_searches = (
        df_weekly_g[df_weekly_g["platform"] == "amazon"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"searchVolume": "sum"})
        .reset_index()
        .rename(columns={"searchVolume": "amazon_searchVolume"})
    )
    chewy_searches = (
        df_weekly_g[df_weekly_g["platform"] == "chewy"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"searchVolume": "sum"})
        .reset_index()
        .rename(columns={"searchVolume": "chewy_searchVolume"})
    )
    google_searches = (
        df_weekly_g[df_weekly_g["platform"] == "google"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"searchVolume": "sum"})
        .reset_index()
        .rename(columns={"searchVolume": "google_searchVolume"})
    )
    walmart_searches = (
        df_weekly_g[df_weekly_g["platform"] == "walmart"]
        .groupby(["date", "Claim_Id", "Claim_name"])
        .agg({"searchVolume": "sum"})
        .reset_index()
        .rename(columns={"searchVolume": "walmart_searchVolume"})
    )

    # Merging all the platform wise searches together in a single dataframe.
    am_ch = amazon_searches.merge(
        chewy_searches, on=["date", "Claim_Id", "Claim_name"], how="outer"
    )
    am_ch_go = am_ch.merge(
        google_searches, on=["date", "Claim_Id", "Claim_name"], how="outer"
    )
    google_search_data_processed = am_ch_go.merge(
        walmart_searches, on=["date", "Claim_Id", "Claim_name"], how="outer"
    )
    google_search_data_processed.fillna(0, inplace=True)

    save_dataset(context, google_search_data_processed, output_dataset)
    return google_search_data_processed


@register_processor("data-cleaning", "sales-social-google")
def process_sales_social_google_data_processed_table(context, params):
    """Prepare the ``SALES SOCIAL GOOGLE Processed `` data table.

    The table is a summary table obtained by doing a ``inner`` join of the
    ```SALES``` data with ```SOCIAL MEDIA```  and again performing ```inner```
    join of the ``SALES SOCIAL`` and ``GOOGLE SEARCH`` tables.
    Further found the top 3 themes and filtered the prepared dataset
    for training.
    """
    input_sales_ds = "processed/sales_data_processed"
    input_social_ds = "processed/social_media_data_processed"
    input_google_ds = "processed/google_search_data_processed"

    output_dataset = "processed/sales_social_google_data_processed"

    # load datasets
    sales_data_processed = load_dataset(context, input_sales_ds)
    social_media_data_processed = load_dataset(context, input_social_ds)
    google_search_data_processed = load_dataset(context, input_google_ds)

    # merging the datasets together
    sales_social_data_processed = sales_data_processed.merge(
        social_media_data_processed, on=["date", "Claim_Id", "Claim_name"], how="inner"
    )

    sales_social_google_data_processed = sales_social_data_processed.merge(
        google_search_data_processed, on=["date", "Claim_Id", "Claim_name"], how="inner"
    )
    sales_social_google_data_processed.drop("date", axis=1, inplace=True)

    top_sales = (
        sales_social_google_data_processed.groupby(["Claim_name"])
        .agg({"client_A_sales_dollars_value": "sum"})
        .reset_index()
        .sort_values(by=["client_A_sales_dollars_value"], ascending=False)
        .head(3)
    )
    sales_social_google_data_processed = sales_social_google_data_processed[
        sales_social_google_data_processed["Claim_name"].isin(top_sales["Claim_name"])
    ]

    save_dataset(context, sales_social_google_data_processed, output_dataset)
    return sales_social_google_data_processed


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES SOCIAL GOOGLE`` table into ``train`` and ``test`` datasets."""

    input_dataset = "processed/sales_social_google_data_processed"
    output_train_features = "train/features"
    output_train_target = "train/target"
    output_test_features = "test/features"
    output_test_target = "test/target"

    # load dataset
    sales_social_google_data_processed = load_dataset(context, input_dataset)

    # split the data
    sales_df_train, sales_df_test = train_test_split(
        sales_social_google_data_processed, test_size=0.2, random_state=42
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        sales_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        sales_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
