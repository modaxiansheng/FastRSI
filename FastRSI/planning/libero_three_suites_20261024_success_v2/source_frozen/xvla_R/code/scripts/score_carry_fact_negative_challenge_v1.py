"""Score frozen fact heads on an exact retained failed-action prefix.

This does not train, grant execution authority, or treat mixed actions as expert
demonstrations. Predictions are saved before challenge facts are scored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from train_carry_fact_head_v1 import CLASSES, SUBJECTS, probabilities, vector


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def features_from_trace(project, trace_path, public_rows, bindings, stop=None):
    rows = [json.loads(s) for s in trace_path.read_text().splitlines()]
    bindings[str(trace_path.relative_to(project))] = sha(trace_path)
    if stop is not None:
        rows = [row for row in rows if row["step"] <= stop]
    if len(rows) != len(public_rows):
        raise ValueError("Public replay alignment differs")
    matrix, metadata, commands = [], [], []
    for i in range(1, len(rows)):
        previous, current = rows[i - 1], rows[i]
        if [previous["step"], current["step"]] != [public_rows[i - 1]["step"], public_rows[i]["step"]]:
            raise ValueError("Replay boundaries differ")
        commands.extend(a[6] for a in current["actions"][:current["step"] - previous["step"]])
        if current["native_success"]:
            continue
        images = []
        for row, public in ((previous, public_rows[i - 1]), (current, public_rows[i])):
            for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                path = trace_path.parent / f'{row["step"]:04d}_{camera}.png'
                digest = sha(path)
                if digest != row["images"][path.name] or digest != public["images"][path.name]:
                    raise ValueError("Source and replay pixels differ")
                bindings[str(path.relative_to(project))] = digest
                with Image.open(path) as image:
                    images.append(image.copy())
        matrix.append(vector(images, public_rows[i]["robot8"], public_rows[i - 1]["robot8"], commands[-20:]))
        metadata.append(dict(boundary=current["step"], replay_index=i))
    return np.stack(matrix), metadata


def main(args):
    started = time.perf_counter()
    registration = read(args.challenge / "REGISTRATION.json")
    replay_path = args.challenge / "job/REPLAY.json"
    replay = read(replay_path)
    if (replay["status"] != "EXACT_IMAGE_REPLAY_RECONSTRUCTED_PUBLIC_STATE"
            or replay["source_actions_are_native_expert"]
            or replay["original_policy_input_state_verified"]):
        raise ValueError("Expected exact mixed-action facts-only challenge")
    args.output.mkdir(exist_ok=False)
    bindings = {str(replay_path.relative_to(args.project)): sha(replay_path),
                str((args.challenge / "REGISTRATION.json").relative_to(args.project)): sha(args.challenge / "REGISTRATION.json"),
                str(Path(__file__).resolve().relative_to(args.project)): sha(Path(__file__).resolve()),
                "scripts/train_carry_fact_head_v1.py": sha(args.project / "scripts/train_carry_fact_head_v1.py")}
    for name, digest in registration["source_manifest"].items():
        if sha(args.project / name) != digest:
            raise ValueError("Registered source changed")
        bindings[name] = digest
    for name, digest in registration["model_manifest"].items():
        if sha(args.project / name) != digest:
            raise ValueError("Frozen model changed")
        bindings[name] = digest

    # Strip training-only physics facts before constructing model inputs.
    public = [{key: row[key] for key in ("step", "images", "robot8")} for row in replay["rows"]]
    save(args.output / "PUBLIC_INPUTS.json", public)
    x, entries = features_from_trace(args.project, args.project / registration["trace"], public,
                                    bindings, registration["stop_step"])
    predictions = {}
    threshold = .95
    for version in ("v1_seed", "v2_updated"):
        predictions[version] = {}
        for subject in SUBJECTS:
            name = f"planning/carry_effect_supervision_v1_20261006/head_job_v1/{version}_{subject}.npy"
            if name not in registration["model_manifest"]:
                raise ValueError("Model absent from pre-challenge registration")
            predictions[version][subject] = probabilities(x, np.load(args.project / name)).tolist()

    # Fixed references receive exactly the v2 learning feedback and legal inputs.
    prep = args.project / "planning/carry_effect_supervision_v1_20261006"
    recipe = read(prep / "HEAD_REGISTRATION.json")
    learning_public = read(prep / "PUBLIC_REPLAY_INPUTS.json")
    learning_labels = read(prep / "LEARNING_SUPERVISION.json")
    sources = read(args.project / "runs/carry_effect_supervision_v1_20261006/REGISTRATION.json")["sources"]
    for name in ("HEAD_REGISTRATION.json", "PUBLIC_REPLAY_INPUTS.json", "LEARNING_SUPERVISION.json"):
        path = prep / name
        bindings[str(path.relative_to(args.project))] = sha(path)
    fixed_matrix, fixed_entries = [], []
    groups = recipe["seed_groups"] + recipe["update_groups"]
    for source in sources:
        if source["init"] not in groups:
            continue
        train_x, meta = features_from_trace(args.project, args.project / source["trace"],
                                           learning_public[str(source["init"])]["rows"], bindings)
        fixed_matrix.extend(train_x)
        fixed_entries.extend(dict(group=source["init"], **row) for row in meta)
    train_x = np.stack(fixed_matrix)
    distances = ((x[:, None] - train_x[None, :]) ** 2).mean(axis=2)
    predictions["fixed_same_evidence_nearest"] = {}
    predictions["fixed_episode_unanimous"] = {}
    for subject in SUBJECTS:
        labels = [learning_labels[f'{row["group"]}:{row["replay_index"]}:{subject}'] for row in fixed_entries]
        predictions["fixed_same_evidence_nearest"][subject] = [labels[k] for k in distances.argmin(axis=1)]
        votes = []
        for group in groups:
            idx = np.asarray([i for i, row in enumerate(fixed_entries) if row["group"] == group])
            votes.append([labels[idx[k]] for k in distances[:, idx].argmin(axis=1)])
        predictions["fixed_episode_unanimous"][subject] = [v[0] if len(set(v)) == 1 else "UNKNOWN" for v in zip(*votes)]
    prediction_path = args.output / "PREDICTIONS.json"
    save(prediction_path, dict(rows=entries, predictions=predictions, model_threshold=threshold,
                               fixed_rule_scores_are_not_probabilities=True))
    save(args.output / "SOURCE_MANIFEST.json", bindings)

    # Only now read challenge labels for scoring, never for fitting or selection.
    truths = {subject: [replay["rows"][row["replay_index"]]["subjects"][subject]["carry_evidence"]["label"]
                        for row in entries] for subject in SUBJECTS}
    metrics, decisions = {}, []
    for version, by_subject in predictions.items():
        metrics[version] = {}
        for subject, values in by_subject.items():
            if version.startswith("v"):
                p = np.asarray(values)
                names = [CLASSES[k] for k in p.argmax(axis=1)]
                confident = p.max(axis=1) >= threshold
                selected = (p.argmax(axis=1) == 1) & confident
            else:
                names = values
                confident = None
                selected = np.asarray([v == CLASSES[1] for v in values])
            y = truths[subject]
            metrics[version][subject] = dict(
                samples=len(y), correct=sum(a == b for a, b in zip(names, y)),
                confident_wrong=(sum(a != b and bool(c) for a, b, c in zip(names, y, confident))
                                 if confident is not None else None),
                supported_true=sum(v == CLASSES[1] for v in y),
                supported_true_selected=sum(bool(c) and v == CLASSES[1] for c, v in zip(selected, y)),
                supported_wrong_selected=sum(bool(c) and v != CLASSES[1] for c, v in zip(selected, y)))
            for i, row in enumerate(entries):
                decisions.append(dict(version=version, subject=subject, boundary=row["boundary"],
                                      predicted=names[i], truth=y[i], supported_selected=bool(selected[i]),
                                      score=(float(max(values[i])) if confident is not None else None)))
    save(args.output / "RESULT.json", dict(
        status="COMPLETED_FROZEN_FAILED_PREFIX_FACT_DIAGNOSTIC", metrics=metrics,
        decisions=decisions, predictions_sha256=sha(prediction_path), full_seconds=time.perf_counter() - started,
        model_trained=False, controller_changed=False, reuse_authorized=False, new_robot_episodes=0,
        original_policy_input_state_verified=False, public_state_reconstructed=True,
        independent_new_context=False,
        scope="Known failed mixed-action init28 prefix, not a third independent context or an action expert. Fact proxy accuracy is not action applicability, task quality, or FastRSI savings."))
    print(json.dumps(dict(status="COMPLETED", metrics=metrics, output=str(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("project", "challenge", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
