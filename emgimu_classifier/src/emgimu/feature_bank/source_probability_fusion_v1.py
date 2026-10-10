"""Source-only convex probability fusion with explicit held-out provenance.

No target labels, per-query fitting or gating network. Personal/calibration
branches remain separately fitted; this policy learns their population mixture.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
import numpy as np
from scipy.optimize import minimize


def explicit_ids(values,name,*,unique=True):
    result=tuple(values)
    if not result or any(not isinstance(t,str) or not t.strip() for t in result) or (unique and len(set(result))!=len(result)):
        raise ValueError('Explicit '+name+' identities required')
    return result


def aligned_probabilities(probabilities,providers,rows,classes):
    if set(probabilities)!=set(providers):raise ValueError('Exact named probability providers required')
    values=[]
    for name in providers:
        q=np.asarray(probabilities[name],float)
        if (q.shape!=(rows,classes) or not np.isfinite(q).all() or np.any(q<0)
                or not np.allclose(q.sum(1),1.,rtol=0,atol=1e-12)):
            raise ValueError('Finite normalized aligned provider probabilities required')
        values.append(q)
    return np.stack(values,axis=1)


@dataclass(frozen=True)
class SourceProbabilityFusionV1:
    classes: tuple
    providers: tuple
    weights: tuple
    source_trial_ids: tuple
    source_protocol_id: str
    inference_bank_id: str
    source_probability_sha256: tuple

    def __post_init__(self):
        for field in ('classes','providers','source_trial_ids'):
            object.__setattr__(self,field,explicit_ids(getattr(self,field),field))
        if len(self.classes)<2:raise ValueError('At least two class identities required')
        w=np.asarray(self.weights,float)
        if w.shape!=(len(self.providers),) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):
            raise ValueError('Nonnegative normalized source weights required')
        object.__setattr__(self,'weights',tuple(float(v) for v in w))
        explicit_ids((self.source_protocol_id,self.inference_bank_id),'source/bank',unique=False)
        hashes=tuple(self.source_probability_sha256)
        if len(hashes)!=len(self.providers) or any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('Source probability hashes required')
        object.__setattr__(self,'source_probability_sha256',hashes)

    @property
    def policy_id(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def manifest(self):return dict(schema='source_probability_fusion_v1',policy_id=self.policy_id,**asdict(self))

    @classmethod
    def from_manifest(cls,value):
        fields={k:v for k,v in value.items() if k not in ('schema','policy_id')}
        policy=cls(**fields)
        if value.get('schema')!='source_probability_fusion_v1' or value.get('policy_id')!=policy.policy_id:
            raise ValueError('Source fusion manifest identity differs')
        return policy

    @classmethod
    def fit(cls,probabilities,labels,*,classes,providers,trial_ids,scenario_ids,user_ids,
            source_protocol_id,inference_bank_id,forbidden_trial_ids=(),regularization=1e-6):
        classes=explicit_ids(classes,'class');providers=explicit_ids(providers,'provider')
        trials=explicit_ids(trial_ids,'source trial',unique=False)
        scenarios=explicit_ids(scenario_ids,'source scenario',unique=False)
        users=explicit_ids(user_ids,'source user',unique=False)
        if (len(scenarios)!=len(trials) or len(users)!=len(trials) or len(set(zip(trials,scenarios)))!=len(trials)
                or set(trials)&set(forbidden_trial_ids) or not np.isfinite(regularization) or regularization<0):
            raise ValueError('Disjoint, unique trial/scenario source rows and fixed regularization required')
        labels=tuple(labels)
        if len(labels)!=len(trials) or not set(labels)==set(classes):raise ValueError('Aligned source labels for every class required')
        for t in set(trials):
            positions=[i for i,x in enumerate(trials) if x==t]
            if len({labels[i] for i in positions})!=1 or len({users[i] for i in positions})!=1:
                raise ValueError('Source native trial class/user differs across scenarios')
        p=aligned_probabilities(probabilities,providers,len(trials),len(classes))
        y=np.array([classes.index(c) for c in labels]);true=p[np.arange(len(y)),:,y]
        sample_weights=np.array([1/(len(set(users))*users.count(u)) for u in users])
        def objective(w):
            v=np.maximum(true@w,1e-15)
            return float(-sample_weights@np.log(v)+regularization*(w@w))
        def gradient(w):
            v=np.maximum(true@w,1e-15)
            return -(sample_weights[:,None]*true/v[:,None]).sum(0)+2*regularization*w
        uniform=np.full(len(providers),1/len(providers))
        optimum=minimize(objective,uniform,jac=gradient,method='SLSQP',bounds=[(0.,1.)]*len(providers),
            constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))},
            options={'ftol':1e-12,'maxiter':2000})
        if not optimum.success:raise ValueError('Source simplex optimization failed: '+optimum.message)
        candidates=[np.maximum(optimum.x,0),uniform,*np.eye(len(providers))]
        candidates=[w/w.sum() for w in candidates];weights=min(candidates,key=objective)
        grad=gradient(weights);gap=float(weights@grad-grad.min())
        if gap>1e-6:raise ValueError('Source simplex stationarity check failed')
        hashes=tuple(hashlib.sha256(np.ascontiguousarray(p[:,i]).tobytes()).hexdigest() for i in range(len(providers)))
        policy=cls(classes,providers,tuple(weights),tuple(sorted(set(trials))),source_protocol_id,inference_bank_id,hashes)
        diagnostic=dict(source_rows=len(trials),independent_source_query_trials=len(set(trials)),source_users=len(set(users)),
            repeated_budget_rows_are_not_independent_trials=True,equal_user_weighting=True,
            regularization=float(regularization),objective=objective(weights),source_log_loss=objective(weights)-regularization*(weights@weights),
            gradient=grad.tolist(),simplex_stationarity_gap=gap,optimizer_iterations=int(optimum.nit),
            target_labels_used=False,source_training_loss_is_unbiased_evaluation=False)
        return policy,diagnostic

    def predict(self,probabilities,*,evaluation_trial_ids,provider_trial_ids,provider_classes,inference_bank_id):
        ids=explicit_ids(evaluation_trial_ids,'evaluation trial')
        if inference_bank_id!=self.inference_bank_id or set(ids)&set(self.source_trial_ids):
            raise ValueError('Inference bank or source/evaluation isolation differs')
        names=tuple(g for g in self.providers if g in probabilities)
        if not names or set(probabilities)!=set(names) or set(provider_trial_ids)!=set(names) or set(provider_classes)!=set(names):
            raise ValueError('Declared available providers and exact axes required')
        if any(tuple(provider_trial_ids[g])!=ids or tuple(provider_classes[g])!=self.classes for g in names):
            raise ValueError('Provider trial/class axes differ')
        p=aligned_probabilities(probabilities,names,len(ids),len(self.classes))
        weights=np.array([self.weights[self.providers.index(g)] for g in names]);rejected=weights.sum()<=1e-15
        q=np.full((len(ids),len(self.classes)),1/len(self.classes)) if rejected else np.einsum('ngc,g->nc',p,weights/weights.sum())
        q/=q.sum(1,keepdims=True)
        return dict(probabilities=q,labels=tuple('Unknown' if rejected else self.classes[i] for i in q.argmax(1)),
            rejected=np.full(len(ids),rejected),providers=names,weights=tuple(weights if rejected else weights/weights.sum()),
            policy_id=self.policy_id,trial_ids=ids,classes=self.classes,physical_validation_proven=False,default_promoted=False)
