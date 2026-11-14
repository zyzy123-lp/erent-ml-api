from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="e-Rent Smart Package Recommender")

EVENT_MAP = {"Wedding": 0, "Birthday": 1, "Debut": 2, "Corporate": 3, "Other": 4}

X_train = np.array((
    (0, 25000, 5, 1, 1),
    (0, 15000, 3, 1, 0),
    (1, 10000, 4, 0, 1),
    (1, 5000, 2, 0, 0),
    (2, 20000, 4, 1, 1),
    (3, 18000, 3, 0, 1),
    (4, 8000, 2, 0, 0),
))
y_train = np.array((1, 2, 3, 4, 5, 6, 7))

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

TENT_IDS = {1, 2}
SOUND_IDS = {5, 6}

class RecommendRequest(BaseModel):
    event_type: str = "Other"
    budget: float = 0.0
    cart_equipment_ids: tuple = ()

@app.get("/")
def home():
    return {"status": "online", "service": "e-Rent ML API"}

@app.post("/recommend")
def recommend_packages(req: RecommendRequest):
    event_code = EVENT_MAP.get(req.event_type, 4)
    has_tent = 1 if any(eid in TENT_IDS for eid in req.cart_equipment_ids) else 0
    has_sound = 1 if any(eid in SOUND_IDS for eid in req.cart_equipment_ids) else 0

    features = np.array(((event_code, req.budget, len(req.cart_equipment_ids), has_tent, has_sound),))
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
