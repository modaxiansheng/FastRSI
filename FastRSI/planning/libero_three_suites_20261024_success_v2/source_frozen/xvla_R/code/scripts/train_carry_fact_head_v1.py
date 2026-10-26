"""CPU-only, fixed-recipe carry-fact heads on retained development replay.

This is a condition-recognition probe, not an action-value learner. Probe facts
are read only after both parameter versions and all predictions are saved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

CLASSES = ("NO_CURRENT_CONTACT", "SUPPORTED_MOTION_WITH_CONTACT", "UNKNOWN")
SUBJECTS = ("white_mug", "yellow_white_mug")


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def vector(images, robot, previous_robot, recent_commands):
    """Legal pixels/proprioception/committed commands; no IDs or hidden facts."""
    if len(images)!=4 or np.asarray(robot).shape!=(8,) or np.asarray(previous_robot).shape!=(8,):
        raise ValueError("Previous/current dual RGB and robot8 required")
    pools=[]
    for image in images:
        if image.mode!="RGB" or image.size!=(256,256): raise ValueError("Unchanged policy pixels required")
        a=np.asarray(image,dtype=np.float32)
        pools.append(a.reshape(8,32,8,32,3).mean(axis=(1,3)).ravel()/127.5-1)
    current=np.asarray(robot,dtype=np.float32)
    prior=np.asarray(previous_robot,dtype=np.float32)
    # Fixed physical scales, registered before any probe labels; no fitted cutoffs.
    current=current*np.array([5,5,5,1/3.2,1/3.2,1/3.2,25,25],np.float32)
    delta=(np.asarray(robot,np.float32)-prior)*np.array([20,20,20,2,2,2,25,25],np.float32)
    commands=np.asarray(recent_commands,dtype=np.float32)
    if not commands.size or not np.isfinite(commands).all(): raise ValueError("Committed commands missing")
    result=np.concatenate(pools+[current,delta,np.array([np.mean(commands>0),commands[-1]>0],np.float32)])
    if not np.isfinite(result).all(): raise ValueError("Nonfinite public feature")
    return result.astype(np.float32)


def weights(groups, labels):
    counts=Counter(zip(groups,labels)); classes=set(labels)
    n={c:len({g for g,y in counts if y==c}) for c in classes}
    return np.asarray([1/(len(classes)*n[y]*counts[g,y]) for g,y in zip(groups,labels)],np.float32)


def probabilities(features, w):
    logits=np.c_[features,np.ones(len(features),np.float32)]@w
    logits-=logits.max(axis=1,keepdims=True)
    values=np.exp(logits)
    return values/values.sum(axis=1,keepdims=True)


def fit(features, labels, groups, recipe):
    if set(labels)!=set(range(3)): raise ValueError("All three seed evidence classes required")
    x=np.c_[features,np.ones(len(features),np.float32)]
    w=np.zeros((x.shape[1],3),np.float32)
    target=np.eye(3,dtype=np.float32)[labels]; voting=weights(groups,labels)
    for _ in range(recipe["iterations"]):
        p=probabilities(features,w)
        gradient=x.T@((p-target)*voting[:,None])
        gradient[:-1]+=recipe["l2"]*w[:-1]
        w-=recipe["learning_rate"]*gradient
    return w


def main(args):
    recipe=read(args.registration)
    if (recipe["iterations"],recipe["learning_rate"],recipe["l2"],recipe["confidence_threshold"],recipe["feature_pooling_grid"])!=(600,.2,.01,.95,8):
        raise ValueError("Fixed carry fact fit recipe changed")
    started=time.perf_counter(); args.output.mkdir(exist_ok=False)
    if (set(recipe["seed_groups"])&set(recipe["update_groups"])
            or set(recipe["seed_groups"]+recipe["update_groups"])&set(recipe["probe_groups"])):
        raise ValueError("Episode groups cross training/probe roles")
    public_path=args.registration.parent/"PUBLIC_REPLAY_INPUTS.json"
    learning_path=args.registration.parent/"LEARNING_SUPERVISION.json"
    probe_path=args.registration.parent/"PROBE_SUPERVISION.json"
    public=read(public_path);learning_labels=read(learning_path)
    sources=read(args.replays/"REGISTRATION.json")["sources"]
    matrix=[]; entries=[]; bindings={str(args.registration.relative_to(args.project)):sha(args.registration)}
    for source in sources:
        stream=args.project/source["stream"]
        trace_path=args.project/source["trace"]
        rows=[json.loads(s) for s in trace_path.read_text().splitlines()]
        label_path=args.replays/f'replays/init{source["init"]}/REPLAY.json'
        # Only the replay alignment and public fields are consumed before fit.
        replay=public[str(source["init"])]
        if replay["status"]!="EXACT_IMAGE_AND_PUBLIC_STATE_REPLAY" or replay["source_trace_sha256"]!=sha(trace_path):
            raise ValueError("Exact source alignment not established")
        if len(replay["rows"])!=len(rows): raise ValueError("Replay rows differ")
        for path in (label_path,trace_path,stream/"OS_COST_EVENTS.jsonl",stream/"RESULT.json"):
            bindings[str(path.relative_to(args.project))]=sha(path)
        recent=[]
        for i in range(1,len(rows)):
            entry=rows[i]; prior=rows[i-1]
            recent.extend(a[6] for a in entry["actions"][:entry["step"]-prior["step"]])
            # Terminal has no future action choice. Never train a skip decision there.
            if entry["native_success"]: continue
            images=[]
            for row in (prior,entry):
                for camera in ("agentview_image","robot0_eye_in_hand_image"):
                    p=trace_path.parent/f'{row["step"]:04d}_{camera}.png'
                    digest=sha(p)
                    if digest!=row["images"][p.name]: raise ValueError("Bound image changed")
                    bindings[str(p.relative_to(args.project))]=digest
                    with Image.open(p) as im: images.append(im.copy())
            tick=time.perf_counter()
            features=vector(images,replay["rows"][i]["robot8"],replay["rows"][i-1]["robot8"],recent[-20:])
            matrix.append(features)
            entries.append(dict(group=source["init"],boundary=entry["step"],replay_index=i,
                                source=source["stream"],feature_seconds=time.perf_counter()-tick))
    x=np.stack(matrix); save(args.output/"SOURCE_MANIFEST.json",bindings)
    def labels_for(subject,groups,supervision):
        ys=[]; idx=[]
        for i,row in enumerate(entries):
            if row["group"] not in groups: continue
            fact=supervision[f'{row["group"]}:{row["replay_index"]}:{subject}']
            idx.append(i);ys.append(CLASSES.index(fact))
        return np.asarray(idx),np.asarray(ys)
    versions={}; training_seconds={}; prediction_seconds={}; predictions={}
    probe_indices=np.asarray([i for i,e in enumerate(entries) if e["group"] in recipe["probe_groups"]])
    for name,groups in (("v1_seed",recipe["seed_groups"]),("v2_updated",recipe["seed_groups"]+recipe["update_groups"])):
        versions[name]={};predictions[name]={}
        for subject in SUBJECTS:
            indices,ys=labels_for(subject,groups,learning_labels);tick=time.perf_counter()
            w=fit(x[indices],ys,[entries[i]["group"] for i in indices],recipe)
            training_seconds[name+":"+subject]=time.perf_counter()-tick
            path=args.output/(name+"_"+subject+".npy");np.save(path,w)
            versions[name][subject]=dict(path=path.name,sha256=sha(path),samples=len(indices),
                label_counts=dict(Counter(CLASSES[int(y)] for y in ys)),groups=groups)
            tick=time.perf_counter();p=probabilities(x[probe_indices],w)
            prediction_seconds[name+":"+subject]=time.perf_counter()-tick
            predictions[name][subject]=p.tolist()
    # Same feedback access, fixed nearest-exemplar rule; no learned parameters.
    predictions["fixed_same_evidence_nearest"]={}
    for subject in SUBJECTS:
        idx,ys=labels_for(subject,recipe["seed_groups"]+recipe["update_groups"],learning_labels)
        d=((x[probe_indices,None]-x[None,idx])**2).mean(axis=2)
        predictions["fixed_same_evidence_nearest"][subject]=np.eye(3)[ys[d.argmin(axis=1)]].tolist()
    save(args.output/"VERSIONS.json",versions)
    save(args.output/"PROBE_PREDICTIONS.json",dict(rows=[entries[i] for i in probe_indices],predictions=predictions))
    # Probe labels are deliberately not read by labels_for during training.
    probe_labels=read(probe_path)
    metrics={}
    for name,prediction in predictions.items():
        metrics[name]={}
        for subject,values in prediction.items():
            idx,y=labels_for(subject,recipe["probe_groups"],probe_labels)
            if not np.array_equal(idx,probe_indices): raise ValueError("Probe alignment changed")
            p=np.asarray(values);a=p.argmax(axis=1); confident=p.max(axis=1)>=recipe["confidence_threshold"]
            chosen=(a==1)&confident; actual=y==1
            groups={}
            for group in recipe["probe_groups"]:
                mask=np.array([entries[i]["group"]==group for i in idx])
                groups[str(group)]=dict(samples=int(mask.sum()),correct=int(((a==y)&mask).sum()),
                    confident_wrong=int(((a!=y)&confident&mask).sum()),
                    supported_true=int((actual&mask).sum()),supported_true_selected=int((chosen&actual&mask).sum()),
                    supported_wrong_selected=int((chosen&~actual&mask).sum()))
            metrics[name][subject]=groups
    for path in (public_path,learning_path,probe_path): bindings[str(path.relative_to(args.project))]=sha(path)
    save(args.output/"SOURCE_MANIFEST.json",bindings)
    save(args.output/"RESULT.json",dict(status="COMPLETED_OFFLINE_CONDITION_PROBE",metrics=metrics,
        versions=versions,training_seconds=training_seconds,prediction_seconds=prediction_seconds,
        mean_feature_seconds=sum(e["feature_seconds"] for e in entries)/len(entries),
        feature_rows=len(entries),feature_dimension=int(x.shape[1]),full_seconds=time.perf_counter()-started,
        added_robot_episodes=0,controller_changed=False,reuse_authorized=False,
        learned_action_value=False,primary_quality_preserved_by_live_test=False,
        scope="Five already seen development episodes. Labels describe contact-supported motion, not grasp truth or useful action reuse. Accuracy is not FastRSI net benefit."))
    print(json.dumps(dict(status="COMPLETED_OFFLINE_CONDITION_PROBE",samples=len(entries),output=str(args.output))))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("project","replays","registration","output"):parser.add_argument("--"+key,type=Path,required=True)
    main(parser.parse_args())
