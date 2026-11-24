"""
CSA1717 Assignment 5 CO5 - reference implementation.
Run: python src/assignment5_co5.py --out results
Dependencies: numpy, scikit-learn, matplotlib, joblib
Optional PySpark RDD pipeline: install pyspark and run the function documented below.
All drone/pump/traffic inputs are teaching simulators, not production data.
"""
import argparse, json, time, random
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RULES = [
("vibration_high","bearing_wear",5),("noise_grinding","bearing_wear",4),("temperature_normal","bearing_wear",1),
("pressure_low","seal_leak",4),("fluid_loss","seal_leak",5),("drip_detected","seal_leak",4),
("temperature_high","motor_overheat",5),("current_high","motor_overheat",4),("smell_burning","motor_overheat",5),
("pressure_fluctuation","cavitation",4),("bubble_noise","cavitation",5),("flow_low","cavitation",2),
("voltage_low","power_fault",5),("current_zero","power_fault",5),("motor_not_starting","power_fault",4)]

def add_rule_at_runtime(rule_base, symptom, fault, weight):
    """Add or replace a rule without changing the inference engine."""
    if not symptom or not fault or not (1 <= int(weight) <= 10):
        raise ValueError("Rule requires a symptom, conclusion and weight from 1 to 10")
    updated = [r for r in rule_base if r[0] != symptom]
    updated.append((symptom, fault, int(weight)))
    return updated

def regression_test(rule_base, prior_cases):
    """Return prior-case predictions so changed diagnoses can be reviewed."""
    global RULES
    original = RULES
    try:
        RULES = list(rule_base)
        return [{"case": c["id"], "prediction": diagnose(c["features"])[0]} for c in prior_cases]
    finally:
        RULES = original

def diagnose(features, strategy="specificity"):
    """Forward-chain symptom rules; specificity sums supporting rule weights."""
    scores, fired = {}, []
    for symptom, fault, weight in RULES:
        if symptom in features:
            scores[fault] = scores.get(fault, 0) + weight
            fired.append({"symptom": symptom, "conclusion": fault, "weight": weight})
    if not scores:
        return "unknown", fired
    if strategy == "priority":
        # Priority is rule weight; lexical fault name is the deterministic tie-break.
        candidates = [(w, f) for s, f, w in RULES if s in features]
        best = max(w for w, _ in candidates)
        prediction = sorted(f for w, f in candidates if w == best)[0]
    else:
        prediction = sorted(scores, key=lambda f: (-scores[f], f))[0]
    return prediction, fired

def drone_swarm(seed=42, n=5, fail_rate=0.0, grid_size=14, max_steps=95):
    """Decentralised frontier selection with local occupancy and visited-cell marks."""
    rng = np.random.default_rng(seed)
    blocked = {(int(rng.integers(1, grid_size-1)), int(rng.integers(1, grid_size-1))) for _ in range(12)}
    blocked.discard((0, 0))
    positions = [(0, 0)] * n
    alive = np.ones(n, dtype=bool)
    failed = int(round(n * fail_rate))
    if failed:
        alive[rng.choice(n, size=failed, replace=False)] = False
    visits, covered = {(0, 0): n}, {(0, 0)}
    for step in range(max_steps):
        next_positions = []
        for i, (r, c) in enumerate(positions):
            if not alive[i]:
                next_positions.append((r, c)); continue
            candidates = [(r+dr, c+dc) for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]]
            candidates = [q for q in candidates if 0 <= q[0] < grid_size and 0 <= q[1] < grid_size and q not in blocked]
            if not candidates:
                next_positions.append((r,c)); continue
            def score(q):
                neighbour_density = sum(abs(q[0]-p[0])+abs(q[1]-p[1]) <= 1 for p in positions)
                return 2*visits.get(q,0) + 1.5*neighbour_density + .04*(abs(q[0]-grid_size+1)+abs(q[1]-grid_size+1)) + rng.random()*.1
            next_positions.append(min(candidates, key=score))
        positions = next_positions
        for q in positions:
            covered.add(q); visits[q] = visits.get(q,0)+1
    total_free = grid_size*grid_size-len(blocked)
    return {"coverage_pct":100*len(covered)/total_free,
            "redundancy_pct":100*max(0,(sum(visits.values())-len(covered))/sum(visits.values())),
            "success":len(covered) >= .5*total_free}

def build_intrusion_models(seed=42):
    """Generated binary intrusion-like benchmark; not a public intrusion corpus."""
    X, y = make_classification(n_samples=5000,n_features=20,n_informative=12,n_redundant=4,
                               weights=[.7,.3],flip_y=.04,random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    full = RandomForestClassifier(n_estimators=70,max_depth=None,min_samples_leaf=1,n_jobs=1,random_state=seed).fit(Xtr,ytr)
    edge = RandomForestClassifier(n_estimators=12,max_depth=6,min_samples_leaf=4,max_features="sqrt",n_jobs=1,random_state=seed).fit(Xtr,ytr)
    def metrics(model):
        pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
        return {"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),
                "recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,prob),
                "confusion_matrix":confusion_matrix(yte,pred).tolist()}
    return full, edge, metrics(full), metrics(edge)

INTENTS = {
"list_alerts":["show open alerts","list current alerts","which alerts are open","display active incidents"],
"explain_flag":["why was this flagged","explain this detection","why is this traffic suspicious","show reason for alert"],
"drone_status":["show drone status","where are the drones","check battery levels","drone fleet status"],
"mission_coverage":["show search coverage","how much area is covered","coverage percentage","display explored grid"],
"pump_diagnosis":["diagnose pump fault","what is wrong with pump","pump fault explanation","check pump symptoms"],
"model_metrics":["show model accuracy","display precision recall and f1","what is the model performance","show detection metrics"],
"edge_status":["is edge model within latency","show edge device status","what is model size","check inference time"],
"deployment_policy":["should this run on edge or cloud","explain deployment decision","where is the assistant hosted","cloud or edge recommendation"]}

class RetrievalAssistant:
    def __init__(self, intents=INTENTS, threshold=.22):
        self.threshold=threshold
        self.labels=[k for k, phrases in intents.items() for _ in phrases]
        corpus=[p for phrases in intents.values() for p in phrases]
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words="english").fit(corpus)
        self.matrix=self.vectorizer.transform(corpus)
    def answer_intent(self, utterance):
        scores=cosine_similarity(self.vectorizer.transform([utterance]),self.matrix)[0]
        i=int(np.argmax(scores)); confidence=float(scores[i])
        return (self.labels[i] if confidence >= self.threshold else "out_of_scope"), confidence

def pyspark_sensor_pipeline(input_path, cache=False):
    """
    Optional true PySpark RDD implementation.
    Input text format: sensorId,timestamp,value
    Transformations: map/filter/reduceByKey; actions: takeOrdered/collect.
    reduceByKey performs map-side combine then shuffle. Cache only if the parsed
    RDD is reused by multiple downstream actions; unpersist after use.
    """
    from pyspark import SparkContext
    sc=SparkContext.getOrCreate()
    raw=sc.textFile(input_path)
    readings=raw.map(lambda line: line.split(",")).filter(lambda x: len(x)==3)
    parsed=readings.map(lambda x:(x[0],(float(x[2]),float(x[2]),1)))
    if cache: parsed=parsed.cache()
    # (sum, max, count) associative aggregate by sensorId
    stats=parsed.reduceByKey(lambda a,b:(a[0]+b[0],max(a[1],b[1]),a[2]+b[2]))
    means=stats.mapValues(lambda x:(x[0]/x[2],x[1]))
    top10=means.takeOrdered(10,key=lambda kv:-kv[1][1])
    result=means.collect()
    if cache: parsed.unpersist()
    return result, top10

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    full,edge,fm,em=build_intrusion_models()
    assistant=RetrievalAssistant()
    examples=["show open alerts","why was this flagged","tell me a joke"]
    chatbot=[{"query":q,"prediction":assistant.answer_intent(q)[0],"confidence":assistant.answer_intent(q)[1]} for q in examples]
    sample_cases=[["vibration_high","noise_grinding"],["fluid_loss","drip_detected"],["voltage_low","current_zero"],["unseen_symptom"]]
    expert=[{"features":x,"diagnosis":diagnose(x)[0],"rules_fired":diagnose(x)[1]} for x in sample_cases]
    swarm=[drone_swarm(seed=42+i,n=n,fail_rate=.1) for i,n in enumerate([5,10,25,50])]
    payload={"intrusion_full":fm,"intrusion_edge":em,"chatbot_examples":chatbot,"expert_examples":expert,"swarm_examples":swarm}
    (out/"results.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
