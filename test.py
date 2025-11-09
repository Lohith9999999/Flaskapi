
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# In-memory data store
items = [
    {"id": 1, "name": "apple", "price": 0.5},
    {"id": 2, "name": "banana", "price": 0.3}
]
_next_id = 3

@app.route('/')
def hello():
    return jsonify({"message": "Hello from Flask API"})

@app.route('/items', methods=['GET'])
def list_items():
    return jsonify(items)

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    for item in items:
        if item["id"] == item_id:
            return jsonify(item)
    abort(404, description="Item not found")

@app.route('/items', methods=['POST'])
def create_item():
    global _next_id
    data = request.get_json(force=True, silent=True)
    if not data or "name" not in data or "price" not in data:
        abort(400, description="JSON with 'name' and 'price' required")
    item = {"id": _next_id, "name": data["name"], "price": data["price"]}
    _next_id += 1
    items.append(item)
    return jsonify(item), 201

@app.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.get_json(force=True, silent=True)
    if not data:
        abort(400, description="JSON body required")
    for item in items:
        if item["id"] == item_id:
            item.update({k: data[k] for k in ("name","price") if k in data})
            return jsonify(item)
    abort(404, description="Item not found")

@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    for i, item in enumerate(items):
        if item["id"] == item_id:
            items.pop(i)
            return '', 204
    abort(404, description="Item not found")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
