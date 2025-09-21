from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import pickle
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
from typing import List, Dict, Any
import json
from datetime import datetime
import sqlite3
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cattle Muzzle Recognition API",
    description="API for cattle identification using muzzle biometrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
recognizer = None
MODEL_PATH = "muzzle_recognition_model.pkl"
DATABASE_PATH = "cattle_database.db"

def generate_cattle_id():
    """Generate 12-digit cattle ID similar to Aadhar"""
    # Generate 12 random digits
    cattle_id = ''.join([str(random.randint(0, 9)) for _ in range(12)])
    return cattle_id

def is_cattle_id_exists(cattle_id: str) -> bool:
    """Check if cattle ID already exists in database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM cattle WHERE cattle_id = ?', (cattle_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def get_unique_cattle_id() -> str:
    """Generate unique 12-digit cattle ID"""
    while True:
        cattle_id = generate_cattle_id()
        if not is_cattle_id_exists(cattle_id):
            return cattle_id

class MuzzleFeatureExtractor(nn.Module):
    """Feature extractor class (same as your training code)"""
    def __init__(self, model_name='resnet50', feature_dim=2048):
        super(MuzzleFeatureExtractor, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
            
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            return features

class CattleRecognitionAPI:
    """Main recognition class for API"""
    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.animal_database = {}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize feature extractor
        self.feature_extractor = MuzzleFeatureExtractor('resnet50').to(self.device)
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize database
        self.init_database()
        
        # Load existing cattle into memory
        self.load_existing_cattle()
    
    def init_database(self):
        """Initialize SQLite database for cattle records"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cattle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cattle_id TEXT UNIQUE NOT NULL,
                owner_name TEXT,
                owner_contact TEXT,
                registration_date TEXT,
                breed TEXT,
                age INTEGER,
                features BLOB,
                image_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_cattle_id TEXT,
                matched_cattle_id TEXT,
                confidence REAL,
                verification_date TEXT,
                decision TEXT,
                image_data BLOB
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_existing_cattle(self):
        """Load existing cattle from database into memory"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cattle_id, owner_name, owner_contact, breed, age, features 
                FROM cattle WHERE status = 'active'
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            for row in results:
                cattle_id = row[0]
                features = pickle.loads(row[5]) if row[5] else None
                
                if features is not None:
                    self.animal_database[cattle_id] = {
                        'avg_features': features,
                        'image_paths': [],
                        'metadata': {
                            'cattle_id': cattle_id,
                            'owner_name': row[1],
                            'owner_contact': row[2],
                            'breed': row[3],
                            'age': row[4]
                        }
                    }
            
            logger.info(f"Loaded {len(self.animal_database)} cattle from database")
            
        except Exception as e:
            logger.error(f"Error loading existing cattle: {e}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as f:
                save_data = pickle.load(f)
            # Don't overwrite database from pickle file
            logger.info(f"Model structure loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def extract_features_from_image(self, image: Image.Image):
        """Extract features from PIL Image"""
        try:
            image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def register_cattle(self, cattle_data: Dict, image: Image.Image) -> Dict:
        """Register new cattle in database"""
        try:
            # Generate unique cattle ID if not provided
            if 'cattle_id' not in cattle_data or not cattle_data['cattle_id']:
                cattle_data['cattle_id'] = get_unique_cattle_id()
            
            # Extract features
            features = self.extract_features_from_image(image)
            if features is None:
                return {"success": False, "error": "Failed to extract features"}
            
            # Check for duplicates
            duplicate_check = self.identify_cattle(image, confidence_threshold=0.90)
            if duplicate_check.get('success', False) and duplicate_check['decision'] == 'MATCH_CONFIDENT':
                return {
                    "success": False, 
                    "error": "Cattle already registered",
                    "existing_cattle_id": duplicate_check['top_match'],
                    "confidence": duplicate_check['confidence']
                }
            
            # Store in database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            features_blob = pickle.dumps(features)
            
            cursor.execute('''
                INSERT INTO cattle 
                (cattle_id, owner_name, owner_contact, registration_date, breed, age, features, image_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cattle_data['cattle_id'],
                cattle_data.get('owner_name', ''),
                cattle_data.get('owner_contact', ''),
                datetime.now().isoformat(),
                cattle_data.get('breed', ''),
                cattle_data.get('age', 0),
                features_blob,
                1
            ))
            
            conn.commit()
            conn.close()
            
            # Update in-memory database
            self.animal_database[cattle_data['cattle_id']] = {
                'avg_features': features,
                'image_paths': [],
                'metadata': cattle_data
            }
            
            logger.info(f"Cattle {cattle_data['cattle_id']} registered successfully")
            return {"success": True, "cattle_id": cattle_data['cattle_id']}
            
        except Exception as e:
            logger.error(f"Error registering cattle: {e}")
            return {"success": False, "error": str(e)}
    
    def identify_cattle(self, image: Image.Image, confidence_threshold: float = 0.85) -> Dict:
        """Identify cattle from image with smart analysis"""
        try:
            # Extract features
            query_features = self.extract_features_from_image(image)
            if query_features is None:
                return {"success": False, "error": "Failed to extract features"}
            
            if not self.animal_database:
                return {"success": False, "error": "No cattle registered in database"}
            
            # Compare with database
            similarities = {}
            for cattle_id, data in self.animal_database.items():
                if 'avg_features' in data:
                    similarity = cosine_similarity(
                        query_features.reshape(1, -1),
                        data['avg_features'].reshape(1, -1)
                    )[0, 0]
                    similarities[cattle_id] = similarity
            
            # Sort by similarity
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Prepare results
            results = []
            for i, (cattle_id, similarity) in enumerate(sorted_matches[:5]):
                results.append({
                    'rank': i + 1,
                    'cattle_id': cattle_id,
                    'similarity': similarity,
                    'confidence': similarity * 100
                })
            
            # Smart decision making
            top_confidence = results[0]['confidence'] / 100
            similarity_gap = 0
            if len(results) > 1:
                similarity_gap = (results[0]['confidence'] - results[1]['confidence']) / 100
            
            # Decision logic
            if top_confidence >= confidence_threshold:
                if similarity_gap >= 0.05:
                    decision = 'MATCH_CONFIDENT'
                    message = f'HIGH CONFIDENCE: This is cattle {results[0]["cattle_id"]}'
                else:
                    decision = 'MATCH_UNCERTAIN'
                    message = f'UNCERTAIN: Could be cattle {results[0]["cattle_id"]}'
            else:
                if top_confidence >= 0.70:
                    decision = 'SIMILAR_NEW'
                    message = 'LIKELY NEW CATTLE: Similar but probably different'
                else:
                    decision = 'NEW_UNIQUE'
                    message = 'NEW UNIQUE CATTLE: Very different muzzle pattern'
            
            return {
                'success': True,
                'decision': decision,
                'message': message,
                'top_match': results[0]['cattle_id'],
                'confidence': top_confidence,
                'similarity_gap': similarity_gap,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Error identifying cattle: {e}")
            return {"success": False, "error": str(e)}
    
    def get_cattle_info(self, cattle_id: str) -> Dict:
        """Get detailed information about specific cattle"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM cattle WHERE cattle_id = ?
            ''', (cattle_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                columns = ['id', 'cattle_id', 'owner_name', 'owner_contact', 
                          'registration_date', 'breed', 'age', 'features', 'image_count', 'status']
                cattle_info = dict(zip(columns, result))
                # Remove features blob from response
                del cattle_info['features']
                return {"success": True, "cattle_info": cattle_info}
            else:
                return {"success": False, "error": "Cattle not found"}
                
        except Exception as e:
            logger.error(f"Error getting cattle info: {e}")
            return {"success": False, "error": str(e)}

# Initialize global recognizer
@app.on_event("startup")
async def startup_event():
    global recognizer
    recognizer = CattleRecognitionAPI()
    logger.info("Cattle Recognition API started successfully")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Cattle Muzzle Recognition API", "status": "active"}

@app.get("/health")
async def health_check():
    cattle_count = len(recognizer.animal_database) if recognizer else 0
    return {
        "status": "healthy",
        "registered_cattle": cattle_count,
        "model_loaded": recognizer is not None
    }

@app.post("/register")
async def register_cattle(
    owner_name: str = Form(...),
    owner_contact: str = Form(...),
    breed: str = Form(...),
    age: int = Form(...),
    file: UploadFile = File(...)
):
    """Register new cattle with muzzle image"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare cattle data
        cattle_data = {
            "owner_name": owner_name,
            "owner_contact": owner_contact,
            "breed": breed,
            "age": age,
        }
        
        # Register cattle
        result = recognizer.register_cattle(cattle_data, image)
        
        if result.get("success"):
            return JSONResponse(content=result, status_code=201)
        else:
            return JSONResponse(content=result, status_code=400)

    except Exception as e:
        logger.error(f"Error in register endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_cattle(file: UploadFile = File(...)):
    """Verify/identify cattle from muzzle image"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Identify cattle
        result = recognizer.identify_cattle(image)

        # Convert NumPy floats to Python floats safely
        if 'confidence' in result:
            result['confidence'] = float(result['confidence'])
        if 'similarity_gap' in result:
            result['similarity_gap'] = float(result['similarity_gap'])
        if 'all_results' in result and isinstance(result['all_results'], list):
            for r in result['all_results']:
                if 'similarity' in r:
                    r['similarity'] = float(r['similarity'])
                if 'confidence' in r:
                    r['confidence'] = float(r['confidence'])

        # If match found, get detailed cattle info
        if result.get('success', False) and result.get('decision') in ['MATCH_CONFIDENT', 'MATCH_UNCERTAIN']:
            cattle_info = recognizer.get_cattle_info(result['top_match'])
            if cattle_info.get('success'):
                result['cattle_details'] = cattle_info['cattle_info']

        # Log verification attempt
        if result.get('success', False):
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO verification_logs 
                (matched_cattle_id, confidence, verification_date, decision, image_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                result.get('top_match', ''),
                result.get('confidence', 0),
                datetime.now().isoformat(),
                result.get('decision', ''),
                image_data
            ))
            
            conn.commit()
            conn.close()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in verify endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cattle/{cattle_id}")
async def get_cattle_details(cattle_id: str):
    """Get detailed information about specific cattle"""
    result = recognizer.get_cattle_info(cattle_id)
    
    if result['success']:
        return JSONResponse(content=result)
    else:
        return JSONResponse(content=result, status_code=404)

@app.get("/cattle")
async def list_all_cattle():
    """Get list of all registered cattle"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cattle_id, owner_name, breed, registration_date, status, owner_contact, age 
            FROM cattle WHERE status = 'active'
            ORDER BY registration_date DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        cattle_list = []
        for row in results:
            cattle_list.append({
                'cattle_id': row[0],
                'owner_name': row[1],
                'breed': row[2],
                'registration_date': row[3],
                'status': row[4],
                'owner_contact': row[5],
                'age': row[6]
            })
        
        return {"success": True, "cattle_count": len(cattle_list), "cattle": cattle_list}
        
    except Exception as e:
        logger.error(f"Error listing cattle: {e}")
        return {"success": False, "error": str(e)}

@app.get("/verification-logs")
async def get_verification_logs(limit: int = 50):
    """Get recent verification logs"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT matched_cattle_id, confidence, verification_date, decision 
            FROM verification_logs 
            ORDER BY verification_date DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in results:
            logs.append({
                'matched_cattle_id': row[0],
                'confidence': row[1],
                'verification_date': row[2],
                'decision': row[3]
            })
        
        return {"success": True, "logs": logs}
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)