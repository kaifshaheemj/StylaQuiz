import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Initialize Flask app
app = Flask(__name__)

# Set HuggingFace cache to local directory
local_dir = r"E:\StylaQuiz\colqwen2_model"
os.environ["HF_HOME"] = local_dir
os.environ["TRANSFORMERS_CACHE"] = local_dir
os.environ["HF_HUB_CACHE"] = local_dir
os.environ["HF_DATASETS_CACHE"] = local_dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load model and processor
try:
    model = ColQwen2.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
        cache_dir=local_dir,
        trust_remote_code=True
    ).eval()

    processor = ColQwen2Processor.from_pretrained(
        local_dir,
        local_files_only=True,
        cache_dir=local_dir,
        trust_remote_code=True
    )
    print("✅ Model and processor loaded successfully!")
    print(f"   📦 Model device: {model.device}")
    print(f"   🔢 Model dtype: {model.dtype}")
except Exception as e:
    print(f"❌ Loading failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Ensure all model files are present in E:\\StylaQuiz\\colqwen2_model")
    print("   2. Verify preprocessor_config.json has valid size keys (shortest_edge, longest_edge)")
    raise Exception("Model loading failed")

# Function to embed queries
def batch_embed_query(query_batch, model_processor, model):
    """Embed a batch of queries and return the embeddings"""
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(model.device)
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()

# API route for batch query embedding
@app.route("/embed-queries", methods=["POST"])
def embed_queries():
    try:
        # Validate request data
        data = request.get_json()
        if not data or "queries" not in data or not isinstance(data["queries"], list):
            return jsonify({"error": "Invalid input: 'queries' must be a list"}), 400

        # Generate embeddings
        
        embeddings = batch_embed_query(data["queries"], processor, model)
        # Convert numpy array to list for JSON serialization
        embeddings_list = embeddings.tolist()
        return jsonify({"embeddings": embeddings_list})
    except Exception as e:
        return jsonify({"error": f"Error processing queries: {str(e)}"}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_device": str(model.device), "model_dtype": str(model.dtype)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)