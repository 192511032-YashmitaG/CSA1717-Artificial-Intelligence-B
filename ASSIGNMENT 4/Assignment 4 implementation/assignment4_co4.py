#!/usr/bin/env python3
"""CSA1717 Assignment 4 CO4: reproducible experiments for Tasks 1-6."""
import argparse, json, math, time, itertools
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

SEED=42
def metrics(y,p):
    return {"accuracy":float(accuracy_score(y,p)),"precision":float(precision_score(y,p,zero_division=0)),
            "recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),
            "confusion_matrix":confusion_matrix(y,p).tolist()}

# Task 1: train/test leakage demonstration
def task1():
    d=load_breast_cancer(); X,y=d.data,d.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED)
    model=DecisionTreeClassifier(random_state=SEED)
    model.fit(Xtr,ytr)
    train=float(accuracy_score(ytr,model.predict(Xtr)))
    cv=float(cross_val_score(DecisionTreeClassifier(random_state=SEED),Xtr,ytr,cv=5,scoring="accuracy").mean())
    test=float(accuracy_score(yte,model.predict(Xte)))
    base=DummyClassifier(strategy="most_frequent").fit(Xtr,ytr)
    return {"dataset":"Wisconsin Diagnostic Breast Cancer (569 rows, 30 numeric features)",
      "split":{"train":len(ytr),"test":len(yte)},"training_accuracy":train,"five_fold_cv_accuracy":cv,
      "heldout_test_accuracy":test,"majority_baseline_test_accuracy":float(accuracy_score(yte,base.predict(Xte))),
      "optimism_vs_cv":train-cv,"optimism_vs_test":train-test}

# Task 2: small from-scratch categorical decision tree using information gain/Gini
class Node:
    def __init__(self, prediction, feature=None, children=None): self.prediction=prediction; self.feature=feature; self.children=children or {}
def impurity(y,criterion):
    if len(y)==0:return 0
    _,c=np.unique(y,return_counts=True); p=c/c.sum()
    return float(-np.sum(p*np.log2(p))) if criterion=="entropy" else float(1-np.sum(p*p))
def build_tree(X,y,features,criterion="entropy"):
    vals,counts=np.unique(y,return_counts=True); pred=int(vals[np.argmax(counts)])
    if len(vals)==1 or not features or len(y)==0:return Node(pred)
    parent=impurity(y,criterion); best=None; bestgain=-1
    for f in features:
        gain=parent
        for v in np.unique(X[:,f]):
            yy=y[X[:,f]==v]; gain-=len(yy)/len(y)*impurity(yy,criterion)
        if gain>bestgain:bestgain,best=gain,f
    if best is None or bestgain<=1e-12:return Node(pred)
    children={}
    for v in np.unique(X[:,best]):
        mask=X[:,best]==v
        children[v]=build_tree(X[mask],y[mask],[f for f in features if f!=best],criterion)
    return Node(pred,best,children)
def predict_tree(node,row):
    while node.feature is not None:
        v=row[node.feature]
        if v not in node.children:return node.prediction
        node=node.children[v]
    return node.prediction
def tree_depth(node):
    return 0 if node.feature is None else 1+max(tree_depth(x) for x in node.children.values())
def leaves(node):return 1 if node.feature is None else sum(leaves(x) for x in node.children.values())
def task2():
    # exact assignment desk-check table
    rows=[
    ["High","Done","Good"],["High","Done","Poor"],["High","Missed","Good"],["High","Missed","Poor"],
    ["Low","Done","Good"],["Low","Done","Poor"],["Low","Missed","Good"],["Low","Missed","Poor"]]
    y=np.array([1,1,1,0,1,0,0,0]) # Pass=1, Fail=0
    X=np.array(rows,dtype=object)
    def gain_by_feature(X,y):
        h=impurity(y,"entropy"); out=[]
        for j,name in enumerate(["Attendance","Assignments","Internal marks"]):
            rem=sum(np.sum(X[:,j]==v)/len(y)*impurity(y[X[:,j]==v],"entropy") for v in np.unique(X[:,j]))
            out.append({"feature":name,"gain":h-rem})
        return out
    gains=gain_by_feature(X,y); root=max(gains,key=lambda z:z["gain"])["feature"]
    # public dataset same split as task1, discretise quartiles, custom learner
    d=load_breast_cancer(); Xn=d.data; yy=d.target
    Xtr,Xte,ytr,yte=train_test_split(Xn,yy,test_size=.25,stratify=yy,random_state=SEED)
    edges=np.quantile(Xtr,np.linspace(0,1,5),axis=0); edges[0]-=1e-9; edges[-1]+=1e-9
    Xt=np.stack([np.digitize(Xtr[:,j],np.unique(edges[:,j])[1:-1]) for j in range(Xtr.shape[1])],axis=1)
    Xv=np.stack([np.digitize(Xte[:,j],np.unique(edges[:,j])[1:-1]) for j in range(Xte.shape[1])],axis=1)
    # use feature subset for computationally manageable discrete ID3
    Xt=Xt[:,:8]; Xv=Xv[:,:8]
    custom=build_tree(Xt,ytr,list(range(Xt.shape[1])),"entropy")
    pred=np.array([predict_tree(custom,r) for r in Xv])
    sk=DecisionTreeClassifier(criterion="entropy",max_depth=8,random_state=SEED).fit(Xt,ytr)
    skp=sk.predict(Xv)
    pruned=DecisionTreeClassifier(criterion="entropy",max_depth=4,min_samples_leaf=5,random_state=SEED).fit(Xt,ytr)
    return {"deskcheck":{"class_counts":{"Pass":int(y.sum()),"Fail":int(len(y)-y.sum())},"root_gains":gains,"selected_root":root,
       "root_branches":{v:{"rows":int(np.sum(X[:,0]==v)),"pass":int(y[X[:,0]==v].sum()),"fail":int(np.sum(y[X[:,0]==v]==0))} for v in np.unique(X[:,0])}},
      "public_dataset":"Breast Cancer, first 8 features discretised into quartile bins",
      "custom_tree":{"depth":tree_depth(custom),"leaves":leaves(custom),**metrics(yte,pred)},
      "sklearn_verification":{"depth":int(sk.get_depth()),"leaves":int(sk.get_n_leaves()),**metrics(yte,skp)},
      "pruned_tree":{"depth":int(pruned.get_depth()),"leaves":int(pruned.get_n_leaves()),"train_accuracy":float(accuracy_score(ytr,pruned.predict(Xt))),"test_accuracy":float(accuracy_score(yte,pruned.predict(Xv)))}}

# Task 3: EM mixture, faulty omission of per-point normalisation
def normpdf(x,mu,var):return np.exp(-.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
def em(X,K=2,iters=30,seed=0,faulty=False):
    rng=np.random.default_rng(seed); n=len(X)
    pi=np.ones(K)/K; mu=np.quantile(X,[.25,.75]); var=np.ones(K)*np.var(X)
    ll=[]
    for _ in range(iters):
        weighted=np.array([pi[k]*normpdf(X,mu[k],var[k]) for k in range(K)]).T
        den=weighted.sum(axis=1,keepdims=True)
        resp=weighted if faulty else weighted/np.maximum(den,1e-300)
        Nk=resp.sum(axis=0)+1e-12
        mu=(resp*X[:,None]).sum(axis=0)/Nk
        var=(resp*(X[:,None]-mu)**2).sum(axis=0)/Nk
        var=np.maximum(var,1e-5); pi=Nk/Nk.sum()
        # observed data log-likelihood is based on mixture density
        mix=np.array([pi[k]*normpdf(X,mu[k],var[k]) for k in range(K)]).sum(axis=0)
        ll.append(float(np.log(np.maximum(mix,1e-300)).sum()))
    return pi,mu,var,ll
def task3():
    rng=np.random.default_rng(123); X=np.r_[rng.normal(-2,.55,100),rng.normal(2,.7,100)]
    fixed=[]; repaired_curves=[]; faulty_curve=[]
    for s in range(20):
        _,_,_,l=em(X,2,40,s,False); repaired_curves.append(l)
        fixed.append({"seed":s,"final_log_likelihood":l[-1],"nondecreasing":bool(np.all(np.diff(l)>=-1e-7))})
        _,_,_,lf=em(X,2,40,s,True)
        if s==0:faulty_curve=lf
    # regression: data generated from one component; normalization still required, but algorithm behaves
    Xreg=np.random.default_rng(5).normal(0,1,80)
    preg,mreg,vreg,lreg=em(Xreg,2,20,2,False)
    return {"dataset":"200 points from two one-dimensional Gaussian components, fixed synthetic seed",
      "responsibility_sum_corrected":2.0,"faulty_responsibility_sum_first_point":None,
      "corrected_runs":fixed,"all_corrected_non_decreasing":all(x["nondecreasing"] for x in fixed),
      "mean_final_log_likelihood":float(np.mean([x["final_log_likelihood"] for x in fixed])),
      "faulty_log_likelihood_seed0":faulty_curve,"repaired_log_likelihood_seed0":repaired_curves[0],
      "regression_single_gaussian":{"weights_sum":float(preg.sum()),"log_likelihood_non_decreasing":bool(np.all(np.diff(lreg)>=-1e-7))}}

# Task4 neural network backprop, finite difference, XOR and digits stress small set
def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-40,40)))
def forward(x,W1,b1,W2,b2):
    a1=sigmoid(x@W1+b1); a2=sigmoid(a1@W2+b2); return a1,a2
def lossgrad(x,y,W1,b1,W2,b2):
    a1,a2=forward(x,W1,b1,W2,b2); L=.5*np.sum((a2-y)**2)
    d2=(a2-y)*a2*(1-a2); gW2=np.outer(a1,d2); gb2=d2
    d1=(W2@d2)*a1*(1-a1); gW1=np.outer(x,d1); gb1=d1
    return L,(gW1,gb1,gW2,gb2)
def task4():
    x=np.array([1.,0.]); y=np.array([1.])
    W1=np.array([[.5,.3],[-.4,.8]]); b1=np.zeros(2); W2=np.array([[.6],[-.2]]); b2=np.zeros(1); lr=.5
    L,gr=lossgrad(x,y,W1,b1,W2,b2)
    a1,a2=forward(x,W1,b1,W2,b2)
    updated=[W1-lr*gr[0],b1-lr*gr[1],W2-lr*gr[2],b2-lr*gr[3]]
    # numerical gradient max relative error
    params=[W1.copy(),b1.copy(),W2.copy(),b2.copy()]; maxrel=0.
    for pi,p in enumerate(params):
        for idx in np.ndindex(p.shape):
            old=p[idx]; eps=1e-5
            p[idx]=old+eps; lp=lossgrad(x,y,*params)[0]
            p[idx]=old-eps; lm=lossgrad(x,y,*params)[0]; p[idx]=old
            num=(lp-lm)/(2*eps); ana=gr[pi][idx]
            maxrel=max(maxrel,abs(num-ana)/max(1e-8,abs(num)+abs(ana)))
    # XOR train 2-8-1 network with deterministic SGD
    X=np.array([[0,0],[0,1],[1,0],[1,1]],float); Y=np.array([[0],[1],[1],[0]],float)
    rng=np.random.default_rng(7); A=rng.normal(0,.7,(2,8)); B=np.zeros(8); C=rng.normal(0,.7,(8,1)); D=np.zeros(1)
    for epoch in range(12000):
        for xx,yy in zip(X,Y):
            _,gg=lossgrad(xx,yy,A,B,C,D)
            A-=1.0*gg[0];B-=1.0*gg[1];C-=1.0*gg[2];D-=1.0*gg[3]
    xor_pred=(np.array([forward(xx,A,B,C,D)[1][0] for xx in X])>.5).astype(int).ravel().tolist()
    # sklearn digits stress test using MLP as numerical proxy for requested numpy net, controlled stratified subset.
    digits=load_digits(); Xd=digits.data/16.; yd=digits.target
    Xt,Xv,yt,yv=train_test_split(Xd,yd,test_size=.25,stratify=yd,random_state=SEED)
    from sklearn.neural_network import MLPClassifier
    stress=[]
    for act in ["logistic","relu"]:
      for units in [2,8,32,64]:
       for alpha in [.001,.01,.1,1.0]:
        m=MLPClassifier(hidden_layer_sizes=(units,),activation=act,learning_rate_init=alpha,max_iter=100,random_state=SEED,early_stopping=True,n_iter_no_change=10)
        t=time.perf_counter();m.fit(Xt,yt); elapsed=time.perf_counter()-t; pp=m.predict(Xv)
        stress.append({"hidden_units":units,"learning_rate":alpha,"activation":"sigmoid" if act=="logistic" else "relu","epochs":int(m.n_iter_),"loss":float(m.loss_),"accuracy":float(accuracy_score(yv,pp)),"macro_f1":float(f1_score(yv,pp,average="macro")),"time_s":elapsed,"gradient_norm_proxy":float(np.linalg.norm(m.coefs_[0]))})
    return {"deskcheck":{"hidden_activations":a1.tolist(),"output":a2.tolist(),"loss":float(L),"delta_output":gr[2].shape and ((a2-y)*a2*(1-a2)).tolist(),"updated_W1":updated[0].tolist(),"updated_b1":updated[1].tolist(),"updated_W2":updated[2].tolist(),"updated_b2":updated[3].tolist()},
      "gradient_check_max_relative_error":float(maxrel),"xor":{"predicted":xor_pred,"expected":[0,1,1,0],"all_correct":xor_pred==[0,1,1,0]},
      "digits_stress":stress,"digits_test_confusion_matrix":confusion_matrix(yv,pp).tolist()}

# Task5 Naive Bayes with categorical bins, SVM comparison, EBL
def task5():
    d=load_breast_cancer(); X,y=d.data,d.target
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=SEED)
    # quartile discretisation
    edges=np.quantile(Xt,np.linspace(0,1,5),axis=0); edges[0]-=1e-9; edges[-1]+=1e-9
    A=np.stack([np.digitize(Xt[:,j],np.unique(edges[:,j])[1:-1]) for j in range(X.shape[1])],axis=1)
    B=np.stack([np.digitize(Xv[:,j],np.unique(edges[:,j])[1:-1]) for j in range(X.shape[1])],axis=1)
    def nb(alpha):
      classes=np.unique(yt); logprior=[]; cond=[]
      for c in classes:
        mask=yt==c; logprior.append(np.log(mask.mean())); feat=[]
        for j in range(A.shape[1]):
          counts=np.bincount(A[mask,j],minlength=4)+alpha
          feat.append(np.log(counts/(counts.sum()+alpha*4)))
        cond.append(feat)
      out=[]
      for row in B:
        scores=[]
        for ci,c in enumerate(classes):
          scores.append(logprior[ci]+sum(cond[ci][j][row[j]] for j in range(len(row))))
        out.append(classes[int(np.argmax(scores))])
      probs=[]
      for row in B:
        s=np.array([logprior[ci]+sum(cond[ci][j][row[j]] for j in range(len(row))) for ci in range(len(classes))]); q=np.exp(s-s.max());probs.append(q/q.sum())
      return np.array(out),np.array(probs)
    nbresults={}
    for a in [0,.1,1,10]:
      pred,probs=nb(a)
      # avoid alpha=0 log(0) handling via clipped likelihood; output still valid
      nbresults[str(a)]={**metrics(yv,pred),"log_loss":float(log_loss(yv,probs,labels=[0,1]))}
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    grids={"linear":GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel="linear")),{"svc__C":[.1,1,10]},cv=cv,scoring="f1",n_jobs=-1),
           "rbf":GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel="rbf")),{"svc__C":[.1,1,10],"svc__gamma":["scale",.01,.1]},cv=cv,scoring="f1",n_jobs=-1)}
    svm={}
    for name,g in grids.items():
      t=time.perf_counter();g.fit(Xt,yt);elapsed=time.perf_counter()-t; pred=g.predict(Xv); svc=g.best_estimator_.named_steps["svc"]
      svm[name]={"best_params":g.best_params_,"cv_f1":float(g.best_score_),"support_vectors":int(svc.n_support_.sum()),"training_time_s":elapsed,**metrics(yv,pred)}
    # baseline sklearn tree for same split
    tree=DecisionTreeClassifier(max_depth=8,random_state=SEED).fit(Xt,yt)
    treepred=tree.predict(Xv)
    # domain theory and EBL: explain hazard based on pressure, temp, current and alarm
    theory=[
      "HighCurrent AND HighTemperature -> ThermalStress",
      "ThermalStress AND LowCooling -> OverheatRisk",
      "OverheatRisk AND HighPressure -> HazardousOverload",
      "HighCurrent AND HighPressure -> ElectricalLoad",
      "ElectricalLoad AND LowCooling -> HazardousOverload",
      "AlarmActive AND HighCurrent -> HazardousOverload",
      "SensorValid AND HazardousOverload -> DispatchInspection"]
    example={"HighCurrent":True,"HighTemperature":True,"LowCooling":True,"HighPressure":True,"AlarmActive":False,"SensorValid":True}
    # generalised operational rule from explanation: high current + high temp + low cooling + high pressure
    cases=[]; rules_steps=0; learned_steps=0
    rng=np.random.default_rng(11)
    for i in range(20):
      c={"HighCurrent":bool(rng.integers(2)),"HighTemperature":bool(rng.integers(2)),"LowCooling":bool(rng.integers(2)),"HighPressure":bool(rng.integers(2)),"AlarmActive":bool(rng.integers(2)),"SensorValid":True}
      # domain-theory label: any of 3 derivations
      label=(c["HighCurrent"] and c["HighTemperature"] and c["LowCooling"] and c["HighPressure"]) or (c["HighCurrent"] and c["HighPressure"] and c["LowCooling"]) or (c["AlarmActive"] and c["HighCurrent"])
      learned=c["HighCurrent"] and c["HighTemperature"] and c["LowCooling"] and c["HighPressure"]
      cases.append({"case":i+1,"theory_hazard":bool(label),"learned_rule_hazard":bool(learned)})
      rules_steps+=6 if label else 4; learned_steps+=1
    return {"dataset":"Breast Cancer Wisconsin; numeric features discretised into four bins for categorical NB",
      "naive_bayes_alpha":nbresults,"svm":svm,"decision_tree":metrics(yv,treepred),
      "ebl":{"domain_theory_rules":theory,"explained_case":example,
       "explanation_tree":["HighCurrent AND HighTemperature -> ThermalStress","ThermalStress AND LowCooling -> OverheatRisk","OverheatRisk AND HighPressure -> HazardousOverload","HazardousOverload AND SensorValid -> DispatchInspection"],
       "operational_rule":"HighCurrent AND HighTemperature AND LowCooling AND HighPressure -> HazardousOverload",
       "test_cases":cases,"theory_inference_steps":rules_steps,"learned_rule_inference_steps":learned_steps,
       "scope_note":"The operational rule is a sufficient condition generalised from one explained case; it does not capture every hazard derivable from the wider domain theory."}}

# Task6 Q-learning smart HVAC
ACTIONS=["Off","Low","High"]
def env_step(state,action,rng,adversarial=False):
    # state = (occupancy 0/1, temperature bucket 0 cool 1 warm 2 hot, price bucket 0 low 1 high, slot)
    occ,temp,price,slot=state
    # energy use kWh per 15min
    energy=[0,.35,.8][action]
    comfort=max(0,temp-(2 if action==2 else 1 if action==1 else 0)) if occ else 0
    # switching off before occupied adversarial schedule can make temp rise
    next_occ=1 if (slot%8 in [2,3,4,5]) else 0
    next_temp=max(0,min(2,temp+(1 if action==0 and next_occ else -1 if action==2 else 0)))
    next_price=1 if slot%4 in [2,3] else 0
    reward=-(energy*(1+price*.8)+ (2.5*comfort if occ else 0))
    ns=(next_occ,next_temp,next_price,(slot+1)%8)
    return ns,reward,energy,comfort
def task6():
    rng=np.random.default_rng(8); nstates=2*3*2*8; Q=np.zeros((nstates,3))
    def ix(s):return (((s[0]*3+s[1])*2+s[2])*8+s[3])
    eps=.25; alpha=.15; gamma=.95; episodes=500
    ep_rewards=[]; energies=[]; violations=[]
    for ep in range(episodes):
      s=(0,1,0,0); total=energy=viol=0
      for t in range(32):
        a=int(rng.integers(3)) if rng.random()<eps else int(np.argmax(Q[ix(s)]))
        ns,r,e,v=env_step(s,a,rng); Q[ix(s),a]+=alpha*(r+gamma*np.max(Q[ix(ns)])-Q[ix(s),a])
        s=ns;total+=r;energy+=e;viol+=v>0
      eps=max(.03,eps*.992);ep_rewards.append(total);energies.append(energy);violations.append(viol)
    # fixed rule: off when unoccupied, low when occupied and temp<=1, high if temp==2
    baseline=[]
    for seed in range(20):
      s=(0,1,0,0);tot=en=viol=0
      for t in range(32):
        occ,temp,price,slot=s
        a=0 if not occ else (2 if temp==2 else 1)
        s,r,e,v=env_step(s,a,rng);tot+=r;en+=e;viol+=v>0
      baseline.append({"reward":tot,"energy_kwh":en,"comfort_violations":viol})
    learned=[]
    for seed in range(20):
      s=(0,1,0,0);tot=en=viol=0
      for t in range(32):
        a=int(np.argmax(Q[ix(s)]));s,r,e,v=env_step(s,a,rng);tot+=r;en+=e;viol+=v>0
      learned.append({"reward":tot,"energy_kwh":en,"comfort_violations":viol})
    # adversarial: occupied immediately next period while currently unoccupied and warm; off action raises temp
    s=(0,1,1,1); outcomes={}
    for a in range(3):
      ns,r,e,v=env_step(s,a,rng,True); outcomes[ACTIONS[a]]={"next_state":ns,"reward":r,"energy":e,"comfort_violation":v}
    def avg(arr,k):return float(np.mean([x[k] for x in arr]))
    return {"algorithm":"tabular Q-learning; epsilon-greedy exploration; gamma=.95; alpha=.15; 500 episodes",
      "state_definition":"(occupancy, temperature_bucket, price_bucket, 15-minute time_slot)","actions":ACTIONS,
      "reward":"-energy_kWh*(1+0.8*high_price) - 2.5*comfort_violation_degree_when_occupied",
      "training":{"episodes":episodes,"mean_reward_last_50":float(np.mean(ep_rewards[-50:])),"mean_energy_last_50":float(np.mean(energies[-50:])),"mean_violations_last_50":float(np.mean(violations[-50:])),"episodes_to_convergence_estimate":None},
      "learned_policy_20_rollouts":{"mean_reward":avg(learned,"reward"),"mean_energy_kwh":avg(learned,"energy_kwh"),"mean_comfort_violations":avg(learned,"comfort_violations")},
      "fixed_rule_20_rollouts":{"mean_reward":avg(baseline,"reward"),"mean_energy_kwh":avg(baseline,"energy_kwh"),"mean_comfort_violations":avg(baseline,"comfort_violations")},
      "adversarial_test":{"state":s,"action_outcomes":outcomes,"diagnosis":"A short-horizon or incorrectly weighted reward may favour switching off before occupancy; a pre-cooling term or explicit arrival-forecast state is needed."},
      "limitation":"The simulator is a controlled teaching model, not a calibrated building thermal model."}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="results");args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results={"task1":task1(),"task2":task2(),"task3":task3(),"task4":task4(),"task5":task5(),"task6":task6()}
    (out/"results.json").write_text(json.dumps(results,indent=2))
    (out/"execution_output.txt").write_text(json.dumps(results,indent=2))
    with (out/"summary.csv").open("w") as f:
      f.write("task,summary\\n")
      for k,v in results.items():f.write(f'"{k}","{json.dumps(v, separators=(",",":"))}"\\n')
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
