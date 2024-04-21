import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np


# Set the style of seaborn
sns.set(style="whitegrid")

# Specify the file path
file_path = "C:\\Users\\Owner\\Downloads\\Code Files\\files\\All Electronics.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Display all records
print(df)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Remove duplicate rows from df_cleaned
df_cleaned = df_cleaned.drop_duplicates()

# Remove unnecessary columns from df_cleaned
columns_to_drop = ['sub_category']
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Remove currency symbols and convert price columns to numeric in df_cleaned
df_cleaned['discount_price'] = df_cleaned['discount_price'].replace('[\₹,]', '', regex=True).astype(float)
df_cleaned['actual_price'] = df_cleaned['actual_price'].replace('[\₹,]', '', regex=True).astype(float)

# Remove or replace outliers in numeric columns
# For example, using z-score to identify and remove outliers
numeric_columns = ['discount_price', 'actual_price', 'ratings', 'no_of_ratings']
# Uncomment the following line if you want to remove outliers using z-score
# df_cleaned = df_cleaned[(np.abs(zscore(df_cleaned[numeric_columns])) < 3).all(axis=1)]

# Convert numeric columns to the appropriate data type
df_cleaned['ratings'] = pd.to_numeric(df_cleaned['ratings'], errors='coerce')
df_cleaned['no_of_ratings'] = pd.to_numeric(df_cleaned['no_of_ratings'], errors='coerce')

print(df_cleaned)


# Function to extract keywords from a text
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return list(set(words))  # Convert to set to get unique keywords

# Apply the keyword extraction function to the 'name' column
df_cleaned['keywords'] = df_cleaned['name'].apply(extract_keywords)

print("\nKeywords Column:")
print(df_cleaned['keywords'])

# Convert numeric columns to the appropriate data type
numeric_columns = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing values with mean without inplace
df_cleaned['ratings'] = df_cleaned['ratings'].fillna(df_cleaned['ratings'].mean())
df_cleaned['no_of_ratings'] = df_cleaned['no_of_ratings'].fillna(df_cleaned['no_of_ratings'].mean())
df_cleaned['discount_price'] = df_cleaned['discount_price'].fillna(df_cleaned['discount_price'].mean())
df_cleaned['actual_price'] = df_cleaned['actual_price'].fillna(df_cleaned['actual_price'].mean())

# Print null count of each column
print(df_cleaned.isnull().sum())

print("DataFrame Length:", len(df_cleaned))

print(df_cleaned.info())
print(df_cleaned.describe())

# Descriptive statistics for numeric columns
numeric_columns = ['ratings', 'no_of_ratings']
numeric_data = df_cleaned[numeric_columns].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors='coerce')

# Plot histograms for each numeric column
plt.figure(figsize=(15, 8))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(numeric_data[column].dropna(), kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Select the top 10 products with max ratings
top_10_products = df_cleaned.nlargest(10, 'ratings')
print("top_10_products : ", top_10_products['name'])

# Select the top 10 products with max discount
top_10_max_discount = df_cleaned.nlargest(10, 'discount_price')

# Select the top 10 products with least discount
top_10_least_discount = df_cleaned.nsmallest(10, 'discount_price')

# Plot horizontal bar chart for top 10 products with max discount
plt.figure(figsize=(12, 6))
sns.barplot(x='discount_price', y='name', data=top_10_max_discount, palette='viridis')
plt.xlabel('Discount Price')
plt.ylabel('Product Name')
plt.title('Top 10 Products with Maximum Discount')
plt.tight_layout()
plt.savefig('top_10_max_discount.png')  # Save the figure as an image
plt.show()

# Plot horizontal bar chart for top 10 products with least discount
plt.figure(figsize=(12, 6))
sns.barplot(x='discount_price', y='name', data=top_10_least_discount, palette='viridis')
plt.xlabel('Discount Price')
plt.ylabel('Product Name')
plt.title('Top 10 Products with Least Discount')
plt.tight_layout()
plt.savefig('top_10_least_discount.png')  # Save the figure as an image
plt.show()


