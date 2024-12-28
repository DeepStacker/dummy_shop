from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Global variables for recommendation system
product_features = None
similarity_matrix = None

# Load the dataset and prepare recommendation system
def load_data():
    try:
        # Load the dataset with the correct column names
        df = pd.read_csv('datab.csv', encoding='latin1')
        
        # Calculate discount percentage
        df['discount'] = ((df['market_price'] - df['sale_price']) / df['market_price'] * 100).round(2)
        
        # Add an ID column if not present
        if 'id' not in df.columns:
            df['id'] = df.index
            
        # Clean up data
        df = df.fillna('')
        
        # Prepare data for recommendation system
        prepare_recommendation_system(df)
        
        print(f"Loaded {len(df)} products")
        print("Columns:", df.columns.tolist())
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def prepare_recommendation_system(df):
    global product_features, similarity_matrix
    
    # Create a combined features string for each product
    df['combined_features'] = df.apply(
        lambda row: f"{row['category']} {row['sub_category']} {row['brand']} {row['type']} {row['description']}", 
        axis=1
    )
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    product_features = tfidf.fit_transform(df['combined_features'])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(product_features)
    
    print("Recommendation system prepared successfully")

def get_similar_products(product_id, n=4):
    try:
        # Get the index of the product in the dataframe
        product_idx = df[df['id'] == product_id].index[0]
        
        # Get similarity scores for this product
        similarity_scores = list(enumerate(similarity_matrix[product_idx]))
        
        # Sort products by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar products (excluding the product itself)
        similar_products_indices = [i[0] for i in similarity_scores[1:n+1]]
        
        # Return the similar products
        return df.iloc[similar_products_indices].to_dict('records')
    except Exception as e:
        print(f"Error in get_similar_products: {e}")
        return []

df = load_data()

# Initialize an empty cart
cart = {
    'items': [],
    'subtotal': 0
}

@app.route('/products', methods=['GET'])
def get_products():
    try:
        # Get filter parameters
        category = request.args.get('category', '').lower()
        brand = request.args.get('brand', '').lower()
        sort = request.args.get('sort', '')
        
        # Apply filters
        filtered_df = df.copy()
        if category and category != 'all categories':
            filtered_df = filtered_df[filtered_df['category'].str.lower() == category]
        if brand and brand != 'all brands':
            filtered_df = filtered_df[filtered_df['brand'].str.lower() == brand]
            
        # Apply sorting
        if sort == 'price_asc':
            filtered_df = filtered_df.sort_values('sale_price')
        elif sort == 'price_desc':
            filtered_df = filtered_df.sort_values('sale_price', ascending=False)
        elif sort == 'rating_desc':
            filtered_df = filtered_df.sort_values('rating', ascending=False)
            
        # Convert to list of dictionaries and limit to 50 products
        products = filtered_df.head(50).to_dict('records')
        
        print(f"Returning {len(products)} products")
        return jsonify(products)
    except Exception as e:
        print(f"Error in get_products: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['GET'])
def search_products():
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify([])
            
        # Search in relevant columns
        search_result = df[
            df['product'].str.lower().str.contains(query, na=False) |
            df['brand'].str.lower().str.contains(query, na=False) |
            df['category'].str.lower().str.contains(query, na=False) |
            df['description'].str.lower().str.contains(query, na=False)
        ]
        
        products = search_result.head(50).to_dict('records')
        return jsonify(products)
    except Exception as e:
        print(f"Error in search_products: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/suggestions', methods=['GET'])
def get_search_suggestions():
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify([])
            
        # Get suggestions from different fields
        product_suggestions = df[df['product'].str.lower().str.contains(query, na=False)]['product'].unique()
        brand_suggestions = df[df['brand'].str.lower().str.contains(query, na=False)]['brand'].unique()
        category_suggestions = df[df['category'].str.lower().str.contains(query, na=False)]['category'].unique()
        
        # Combine and format suggestions
        suggestions = []
        
        # Add product name suggestions
        for product in product_suggestions[:3]:
            suggestions.append({
                'type': 'product',
                'text': product,
                'icon': '🏷️'
            })
            
        # Add brand suggestions
        for brand in brand_suggestions[:2]:
            suggestions.append({
                'type': 'brand',
                'text': brand,
                'icon': '®️'
            })
            
        # Add category suggestions
        for category in category_suggestions[:2]:
            suggestions.append({
                'type': 'category',
                'text': category,
                'icon': '📁'
            })
            
        # Get recommended products based on query
        recommended_products = df[
            df['product'].str.lower().str.contains(query, na=False) |
            df['brand'].str.lower().str.contains(query, na=False) |
            df['category'].str.lower().str.contains(query, na=False)
        ].head(3).to_dict('records')
        
        return jsonify({
            'suggestions': suggestions[:5],  # Limit to top 5 suggestions
            'recommended_products': recommended_products
        })
    except Exception as e:
        print(f"Error in get_search_suggestions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cart', methods=['GET'])
def get_cart():
    return jsonify(cart)

@app.route('/cart/add', methods=['POST'])
def add_to_cart():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        
        if product_id is None:
            return jsonify({'error': 'Product ID is required'}), 400
            
        # Find the product in the dataset
        product = df[df['id'] == product_id]
        if product.empty:
            return jsonify({'error': 'Product not found'}), 404
            
        product = product.iloc[0]
        
        # Check if product already exists in cart
        existing_item = next((item for item in cart['items'] if item['id'] == product_id), None)
        
        if existing_item:
            existing_item['quantity'] += 1
        else:
            cart['items'].append({
                'id': product_id,
                'product': product['product'],  # Using 'product' as the name
                'brand': product['brand'],
                'price': float(product['sale_price']),
                'market_price': float(product['market_price']),
                'quantity': 1
            })
            
        # Update subtotal
        cart['subtotal'] = sum(item['price'] * item['quantity'] for item in cart['items'])
        
        return jsonify(cart)
    except Exception as e:
        print(f"Error in add_to_cart: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cart/remove', methods=['POST'])
def remove_from_cart():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        
        if product_id is None:
            return jsonify({'error': 'Product ID is required'}), 400
        
        # Remove item from cart
        cart['items'] = [item for item in cart['items'] if item['id'] != product_id]
        
        # Update subtotal
        cart['subtotal'] = sum(item['price'] * item['quantity'] for item in cart['items'])
        
        return jsonify(cart)
    except Exception as e:
        print(f"Error in remove_from_cart: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cart/update', methods=['POST'])
def update_cart_quantity():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity = data.get('quantity')
        
        if product_id is None or quantity is None:
            return jsonify({'error': 'Product ID and quantity are required'}), 400
        
        # Update item quantity
        for item in cart['items']:
            if item['id'] == product_id:
                item['quantity'] = quantity
                break
                
        # Update subtotal
        cart['subtotal'] = sum(item['price'] * item['quantity'] for item in cart['items'])
        
        return jsonify(cart)
    except Exception as e:
        print(f"Error in update_cart_quantity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/product/<int:product_id>', methods=['GET'])
def get_product_detail(product_id):
    try:
        # Find the product
        product = df[df['id'] == product_id]
        if product.empty:
            return jsonify({'error': 'Product not found'}), 404
            
        product_data = product.iloc[0].to_dict()
        
        # Get similar products using recommendation system
        similar_products = get_similar_products(product_id)
        
        return jsonify({
            'product': product_data,
            'similar_products': similar_products
        })
    except Exception as e:
        print(f"Error in get_product_detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    try:
        # Get unique categories with their subcategories and product counts
        categories = []
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            subcategories = category_df['sub_category'].unique().tolist()
            product_count = len(category_df)
            
            # Get a sample product for the category image (in real app, you'd have category images)
            sample_product = category_df.iloc[0]
            
            categories.append({
                'name': category,
                'subcategories': subcategories,
                'productCount': product_count,
                'sampleProduct': {
                    'name': sample_product['product'],
                    'price': sample_product['sale_price']
                }
            })
            
        return jsonify(categories)
    except Exception as e:
        print(f"Error in get_categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/featured', methods=['GET'])
def get_featured():
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df_featured = df.copy()
        
        # Convert price columns to numeric
        df_featured['market_price'] = pd.to_numeric(df_featured['market_price'], errors='coerce')
        df_featured['sale_price'] = pd.to_numeric(df_featured['sale_price'], errors='coerce')
        
        # Calculate discount amount
        df_featured['discount_amount'] = df_featured['market_price'] - df_featured['sale_price']
        
        # Function to safely convert records to JSON-compatible format
        def clean_records(records):
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        cleaned_record[key] = float(value) if isinstance(value, np.floating) else int(value)
                    elif pd.isna(value):
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            return cleaned_records

        # Get top rated products
        try:
            # Try to convert ratings to numeric
            df_featured['rating_num'] = pd.to_numeric(df_featured['rating'], errors='coerce')
            top_rated = df_featured[df_featured['rating_num'].notna()].nlargest(4, 'rating_num')
            top_rated = clean_records(top_rated.to_dict('records'))
        except Exception as e:
            print(f"Error getting top rated products: {e}")
            # Fallback: get random products
            top_rated = clean_records(df_featured.sample(n=min(4, len(df_featured))).to_dict('records'))

        # Get best deals
        try:
            best_deals = df_featured[
                (df_featured['discount_amount'].notna()) & 
                (df_featured['discount_amount'] > 0)
            ].nlargest(4, 'discount_amount')
            best_deals = clean_records(best_deals.to_dict('records'))
        except Exception as e:
            print(f"Error getting best deals: {e}")
            # Fallback: get random products
            best_deals = clean_records(df_featured.sample(n=min(4, len(df_featured))).to_dict('records'))

        # Get new arrivals (random for demo)
        new_arrivals = clean_records(df_featured.sample(n=min(4, len(df_featured))).to_dict('records'))
        
        # Get trending (random for demo)
        trending = clean_records(df_featured.sample(n=min(4, len(df_featured))).to_dict('records'))

        return jsonify({
            'topRated': top_rated,
            'bestDeals': best_deals,
            'newArrivals': new_arrivals,
            'trending': trending
        })
    except Exception as e:
        print(f"Error in get_featured: {e}")
        # Return empty arrays as fallback
        return jsonify({
            'topRated': [],
            'bestDeals': [],
            'newArrivals': [],
            'trending': [],
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
