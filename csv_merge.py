import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import random
from sklearn.model_selection import *
from sqlalchemy import create_engine
import sklearn
import time
import os.path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


aisles = pd.read_csv('data\\aisles.csv')
products = pd.read_csv('data\\products.csv')
departments = pd.read_csv('data\\departments.csv')

orders = pd.read_csv('data\\orders.csv')

prod_prior = pd.read_csv('data\\order_products__prior.csv')
prod_train = pd.read_csv('data\\order_products__train.csv')

print(prod_prior.head(5))
merged_prior_order = pd.merge(prod_prior, orders , on='order_id')
