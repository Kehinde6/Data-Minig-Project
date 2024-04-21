import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df_cleaned = pd.read_csv("C:\\Users\\Owner\\Downloads\\Code Files\\files\\All Electronics.csv")

# Sample dataset
data = {'name': [
    'Redmi 10 Power (Power Black, 8GB RAM, 128GB Storage)',
    'OnePlus Nord CE 2 Lite 5G (Blue Tide, 6GB RAM, ...)',
    'OnePlus Bullets Z2 Bluetooth Wireless in Ear Earphones',
    'Samsung Galaxy M33 5G (Mystique Green, 6GB, 128GB Storage)',
    'OnePlus Nord CE 2 Lite 5G (Black Dusk, 6GB RAM, ...)',
    'Redmi 10 Power (Sporty Orange, 8GB RAM, 128GB Storage)',
    'boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds',
    'Apple 20W USB-C Power Adapter (for iPhone, iPad)',
    'Fire-Boltt Ninja Call Pro Plus 1.83" Smart Watch',
    'Samsung Galaxy M33 5G (Emerald Brown, 6GB, 128GB Storage)'
],
'category': ['phone', 'phone', 'bluetooth', 'phone', 'phone', 'phone', 'bluetooth', 'adaptor', 'watch', 'phone']
}

df = pd.DataFrame(data)

# Basic setup
X_train, X_test, y_train, y_test = train_test_split(df['name'], df['category'], test_size=0.2, random_state=42)

# Train model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Save model
model_filename = 'product_category_model.joblib'
joblib.dump(model, model_filename)

# Model inference testing
test_product_names = [
    'Sony Noise Cancelling Headphones WH1000XM3',
    'Canon EOS 5D Mark IV Full Frame Digital SLR Camera',
    'Fitbit Charge 4 Fitness and Activity Tracker with Built-in GPS',
    'Samsung Galaxy Watch 4 Classic',
    'Anker PowerCore 10000 Portable Charger'
]

# Load saved model
loaded_model = joblib.load(model_filename)

# Make predictions
predictions = loaded_model.predict(test_product_names)

# Print predictions
print("\nInference Testing:", predictions)
for product, prediction in zip(test_product_names, predictions):
    print(f"Product: {product}, Predicted Category: {prediction}")

# Apply the model on the 'name' column of df_cleaned
df_cleaned['predProdCat'] = loaded_model.predict(df_cleaned['name'])

# Convert 'ratings' column to numeric
df_cleaned['ratings'] = pd.to_numeric(df_cleaned['ratings'], errors='coerce')

# Category Distribution
category_counts = df_cleaned['predProdCat'].value_counts()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Number of Products')
plt.title('Category Distribution')
plt.xticks(rotation=45, ha='right')
plt.show()

# Average Ratings per Category
avg_ratings_per_category = df_cleaned.groupby('predProdCat')['ratings'].mean()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_ratings_per_category.index, y=avg_ratings_per_category.values, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Average Ratings')
plt.title('Average Ratings per Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Discount Price Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='predProdCat', y='discount_price', data=df_cleaned, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Discount Price')
plt.title('Discount Price Distribution per Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Number of Ratings Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='predProdCat', y='no_of_ratings', data=df_cleaned, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Number of Ratings')
plt.title('Number of Ratings Distribution per Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Average Discount Percentage per Category
df_cleaned['discount_percentage'] = ((df_cleaned['actual_price'] - df_cleaned['discount_price']) / df_cleaned['actual_price']) * 100
avg_discount_percentage_per_category = df_cleaned.groupby('predProdCat')['discount_percentage'].mean()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_discount_percentage_per_category.index, y=avg_discount_percentage_per_category.values, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Average Discount Percentage')
plt.title('Average Discount Percentage per Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Average Actual Price per Category
avg_actual_price_per_category = df_cleaned.groupby('predProdCat')['actual_price'].mean()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_actual_price_per_category.index, y=avg_actual_price_per_category.values, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Average Actual Price')
plt.title('Average Actual Price per Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Percentage of Products on Discount per Category
discounted_products_percentage_per_category = (df_cleaned.groupby('predProdCat')['discount_price'].count() / len(df_cleaned)) * 100

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=discounted_products_percentage_per_category.index, y=discounted_products_percentage_per_category.values, palette='viridis')
plt.xlabel('Predicted Categories')
plt.ylabel('Percentage of Products on Discount')
plt.title('Percentage of Products on Discount per Category')
plt.xticks(rotation=45, ha='right')
plt.show()
