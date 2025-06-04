import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

regions = [
    {"Country": "United States", "City": "New York", "State": "New York", "Region": "East"},
    {"Country": "United States", "City": "Los Angeles", "State": "California", "Region": "West"},
    {"Country": "United States", "City": "Chicago", "State": "Illinois", "Region": "Central"},
    {"Country": "Canada", "City": "Toronto", "State": "Ontario", "Region": "Central"},
    {"Country": "Australia", "City": "Sydney", "State": "NSW", "Region": "APAC"},
    {"Country": "United Kingdom", "City": "London", "State": "England", "Region": "EMEA"},
    {"Country": "India", "City": "Mumbai", "State": "Maharashtra", "Region": "APAC"},
    {"Country": "Germany", "City": "Berlin", "State": "Berlin", "Region": "EMEA"},
    {"Country": "Brazil", "City": "São Paulo", "State": "São Paulo", "Region": "LATAM"},
    {"Country": "South Africa", "City": "Cape Town", "State": "Western Cape", "Region": "MEA"},
]

segments = ["Consumer", "Corporate", "Home Office"]
categories = [
    ("Office Supplies", ["Binders", "Paper", "Writing", "Envelopes"]),
    ("Furniture", ["Chairs", "Tables", "Bookcases"]),
    ("Technology", ["Phones", "Accessories", "Monitors", "Tablets"])
]

products = []
for i in range(1, 41):
    cat, subcats = random.choice(categories)
    subcat = random.choice(subcats)
    pname = fake.word().capitalize() + " " + subcat
    pid = f"PROD-{i:03d}"
    products.append({"ProductID": pid, "ProductName": pname, "Category": cat, "Sub-Category": subcat})

def random_order_id(idx, year):
    return f"CA-{year}-{100000+idx}"

def random_date(start_year=2022, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime('%Y-%m-%d')

def generate_sales_data(num_rows=1500):
    orders = []
    for i in range(num_rows):
        year = random.choice([2022, 2023, 2024])
        order_id = random_order_id(i, year)
        order_date = random_date()
        customer_id = f"CUS-{1000 + random.randint(1, 500)}"
        customer_name = fake.name()
        segment = random.choice(segments)
        prod = random.choice(products)
        quantity = random.randint(1, 10)
        unit_price = round(random.uniform(2, 800), 2)
        discount = round(random.choice([0, 0.05, 0.1, 0.12]), 2)
        sales = round(quantity * unit_price * (1 - discount), 2)
        profit = round(sales * random.uniform(0.05, 0.25), 2)
        loc = random.choice(regions)
        orders.append({
            "OrderID": order_id,
            "OrderDate": order_date,
            "CustomerID": customer_id,
            "CustomerName": customer_name,
            "Segment": segment,
            "ProductID": prod["ProductID"],
            "ProductName": prod["ProductName"],
            "Category": prod["Category"],
            "Sub-Category": prod["Sub-Category"],
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Discount": discount,
            "Profit": profit,
            "Sales": sales,
            "Country": loc["Country"],
            "City": loc["City"],
            "State": loc["State"],
            "Region": loc["Region"]
        })
    return pd.DataFrame(orders)

if __name__ == "__main__":
    df = generate_sales_data(1500)
    df.to_csv("sales_data.csv", index=False)
    print("Generated sales_data.csv with", len(df), "rows.")
