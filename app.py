from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="e-Rent Smart Package Recommender")

EVENT_MAP = {"Wedding": 0, "Birthday": 1, "Debut": 2, "Corporate": 3, "Other": 4}

# Sample training data: [event_code, budget, item_count, has_tent, has_sound] -> package_id
X_train = np.array([,  # Wedding -> Package 1,  # Wedding -> Package 2,  # Birthday -> Package 3,  # Birthday -> Package 4,  # Debut -> Package 5,  # Corporate -> Package 6,  # Other -> Package 7
])
y_train = np.array()

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

class RecommendRequest(BaseModel):
    event_type: str = "Other"
    budget: float = 0.0
    cart_equipment_ids: List[int] = []

@app.get("/")
def home():
    return {"status": "online", "service": "e-Rent ML API"}

@app.post("/recommend")
def recommend_packages(req: RecommendRequest):
    event_code = EVENT_MAP.get(req.event_type, 4)
    has_tent = 1 if any(eid in req.cart_equipment_ids for eid in) else 0
    has_sound = 1 if any(eid in req.cart_equipment_ids for eid in) else 0

    features = np.array([[event_code, req.budget, len(req.cart_equipment_ids), has_tent, has_sound]])
    probs = model.predict_proba(features)
    classes = model.classes_

    ranked_indices = np.argsort(probs)[::-1]
    recommendations = [
        {"package_id": int(classes[i]), "score": round(float(probs[i]), 3)}
        for i in ranked_indices if probs[i] > 0.05
    ]

    return {"recommendations": recommendations[:3]}
