from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="e-Rent Smart Package Recommender")

EVENT_MAP = {"Wedding": 0, "Birthday": 1, "Debut": 2, "Corporate": 3, "Other": 4}

# Training features: (event_code, budget, guest_count, item_count, has_tent, has_sound)
X_train = np.array((
    (0, 20000, 50, 4, 1, 0),   # Wedding -> Package 5
    (1, 20000, 50, 3, 1, 0),   # Birthday -> Package 5
    (2, 20000, 60, 4, 1, 0),   # Debut -> Package 5
    (3, 15000, 40, 3, 1, 1),   # Corporate -> Package 6
    (4, 15000, 40, 3, 1, 1),   # Other -> Package 6
    (2, 30000, 100, 5, 0, 0),  # Debut (Large) -> Package 7
    (0, 30000, 120, 5, 0, 0),  # Wedding (Large) -> Package 7
))

# Real package_id values from your erent_db
y_train = np.array((5, 5, 5, 6, 6, 7, 7))

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Real equipment_id values from your erent_db
TENT_IDS = {3, 8}
SOUND_IDS = {4}

class RecommendRequest(BaseModel):
    event_type: str = "Other"
    budget: float = 0.0
    guest_count: int = 0
    cart_equipment_ids: tuple = ()

@app.get("/")
def home():
    return {"status": "online", "service": "e-Rent ML API"}

@app.post("/recommend")
def recommend_packages(req: RecommendRequest):
    event_code = EVENT_MAP.get(req.event_type, 4)
    has_tent = 1 if any(eid in TENT_IDS for eid in req.cart_equipment_ids) else 0
    has_sound = 1 if any(eid in SOUND_IDS for eid in req.cart_equipment_ids) else 0

    features = np.array(((
        event_code,
        req.budget,
        req.guest_count,
        len(req.cart_equipment_ids),
        has_tent,
        has_sound
    ),))

    probs = model.predict_proba(features).flatten()
    classes = model.classes_

    ranked_indices = np.argsort(probs)[::-1]
    recommendations = list()
    for i in ranked_indices:
        if probs.item(i) > 0.05:
            recommendations.append({
                "package_id": int(classes.item(i)),
                "score": round(float(probs.item(i)), 3)
            })

    return {"recommendations": recommendations[:3]}
