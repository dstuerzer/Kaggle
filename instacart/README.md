Data Exploration for the Kaggle Competition "Instacart Market Basket Analysis".

Most successful attempts:

 - `script_xgb` uses additional features such as: current product frequency, time since last product order, ... Drawback: Removes all entries before first occurence of product, since no 'time since last product order' is known.

- `script_xgb_0th` same as other XGBoost-model. However, a guess is made about a 0-th order for a given product, in order to be able to obtain data for 'time since last product order' also for orders before the first product order. The guess was obtained from a data analysis of order patterns of various products.

- `simple_model` Simply looks at the last 10 (at most) orders, analyzes the frequency of products in those orders, and predicts the products for the test set that have occurred with a frequency of more than 0.35 (a value obtained by grid search).
