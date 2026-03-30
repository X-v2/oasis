import requests

def fetch_construction_materials(search_term=None):
    # Use search endpoint if a term is provided, else get all
    if search_term:
        url = f"https://api.konnbot.in/api/products/search/?q={search_term}"
    else:
        url = "https://api.konnbot.in/api/products/"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # If the API returns a list of products
            for product in data:
                name = product.get('name', 'N/A')
                price = product.get('price', 'Call for Price')
                unit = product.get('unit', 'unit')
                print(f"[{product.get('category', 'Material')}] {name}: ₹{price} per {unit}")
        else:
            print(f"API Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")

# Example: Fetching all Steel-related products
print("--- Steel Rates ---")
fetch_construction_materials("Steel")

# Example: Fetching all Cement-related products
print("\n--- Cement Rates ---")
fetch_construction_materials("Cement")