"""Frozen Clef features and two external object-stage eligibility heads.

This is offline, grouped development supervision of native subgoal plus release
facts. It neither executes a robot nor labels counterfactual routing decisions.
Probe supervision is read only after checkpoints and predictions are saved.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LABEL_SCOPE = "native_subgoal_plus_release_two_observed_boundaries"
SUBJECT_NAMES = {
    "white": "white_mug", "white_mug": "white_mug",
    "yellow_white": "yellow_white_mug", "yellow_white_mug": "yellow_white_mug",
}
INPUT_FIELDS = {"id", "group", "role", "subject", "images", "request"}
LABEL_FIELDS = {"id", "group", "role", "subject", "eligible_to_skip", "label_scope"}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha_text(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Missing or malformed SHA256")
    return value


def project_file(project, relative):
    value = Path(relative)
    if not isinstance(relative, str) or value.is_absolute() or ".." in value.parts:
        raise ValueError("Project inputs must use relative paths without traversal")
    resolved = (project / value).resolve()
    if not resolved.is_relative_to(project.resolve()):
        raise ValueError("Project input resolves outside project")
    return resolved


def validate_inputs(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty INPUTS.json list required")
    indexed, group_roles, signatures = {}, {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != INPUT_FIELDS:
            raise ValueError("Unexpected input fields")
        for key in ("id", "group"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError("Missing " + key)
        if row["id"] in indexed or row["role"] not in ("seed", "update", "probe"):
            raise ValueError("Duplicate input ID or invalid role")
        if row["subject"] not in SUBJECT_NAMES:
            raise ValueError("Unknown object subject")
        role = group_roles.setdefault(row["group"], row["role"])
        if role != row["role"]:
            raise ValueError("An episode group crosses learning/probe roles")
        images = row["images"]
        if not isinstance(images, list) or len(images) != 4:
            raise ValueError("Previous/current front/wrist images required")
        for image in images:
            if (not isinstance(image, dict)
                    or set(image) != {"path", "sha256", "pixel_sha256"}
                    or not isinstance(image["path"], str)):
                raise ValueError("Unexpected image reference")
            sha_text(image["sha256"]); sha_text(image["pixel_sha256"])
        request = row["request"]
        if (not isinstance(request, dict) or "images" in request
                or request.get("state", {}).get("subject") != SUBJECT_NAMES[row["subject"]]):
            raise ValueError("Subject/request mismatch or pixels embedded in JSON request")
        signature = hashlib.sha256(json.dumps(
            [request, [image["pixel_sha256"] for image in images]],
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        old = signatures.setdefault(signature, row["role"])
        if old != row["role"]:
            raise ValueError("Identical model input crosses learning/probe roles")
        indexed[row["id"]] = row
    if set(group_roles.values()) != {"seed", "update", "probe"}:
        raise ValueError("Seed, update and grouped probe inputs all required")
    return indexed


def validate_labels(labels, indexed, allowed_roles):
    if not isinstance(labels, list):
        raise ValueError("Supervision must be a list")
    expected = {key for key, row in indexed.items() if row["role"] in allowed_roles}
    result = {}
    for label in labels:
        if not isinstance(label, dict) or set(label) != LABEL_FIELDS:
            raise ValueError("Unexpected supervision fields")
        key = label["id"]
        if key not in expected or key in result:
            raise ValueError("Unknown, duplicate or disallowed supervision ID")
        row = indexed[key]
        if any(label[field] != row[field] for field in ("group", "role", "subject")):
            raise ValueError("Supervision group/role/subject mismatch")
        if (type(label["eligible_to_skip"]) is not bool
                or label["label_scope"] != LABEL_SCOPE):
            raise ValueError("Invalid native object-stage supervision")
        result[key] = label
    if set(result) != expected:
        raise ValueError("Supervision ID coverage mismatch")
    return result


def grouped_class_weights(labels):
    """Each episode has one vote per represented class, independent of frames."""
    counts = Counter((label["group"], label["eligible_to_skip"]) for label in labels)
    classes = {label["eligible_to_skip"] for label in labels}
    class_groups = {value: len({group for group, cls in counts if cls == value})
                    for value in classes}
    return [1.0 / (len(classes) * class_groups[label["eligible_to_skip"]]
                   * counts[(label["group"], label["eligible_to_skip"])])
            for label in labels]


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="VALIDATING", new_robot_episodes=0, controller_changed=False,
                  backbone_updates=0, live_os_model_calls=0, calls=[])

    def save():
        report["full_wall_seconds"] = time.perf_counter() - started
        (args.output / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    try:
        protocol = read_json(args.inputs / "PROTOCOL.json")
        fit = protocol["fit"]
        if (fit.get("seed") != 17 or fit.get("learning_rate") != 0.02
                or fit.get("iterations") != 400 or fit.get("weight_decay") != 0.01
                or protocol.get("threshold") != 0.95):
            raise ValueError("Object-stage v1 fixed fit/threshold protocol changed")
        rows = read_json(args.inputs / "INPUTS.json")
        indexed = validate_inputs(rows)
        labels = validate_labels(read_json(args.inputs / "SUPERVISION.json"),
                                 indexed, {"seed", "update"})
        report["input_artifact_sha256"] = {
            name: file_sha(args.inputs / name) for name in
            ("INPUTS.json", "SUPERVISION.json", "PROTOCOL.json", "SOURCE_MANIFEST.json")}
        for relative, expected in read_json(args.inputs / "SOURCE_MANIFEST.json").items():
            if file_sha(project_file(args.project, relative)) != sha_text(expected):
                raise ValueError("Project source changed: " + relative)
        manifest = read_json(args.model / "DOWNLOAD.json")
        if (manifest.get("status") != "VERIFIED"
                or manifest.get("revision") != protocol.get("model_revision")):
            raise ValueError("Unverified or wrong model revision")
        for name, item in manifest["files"].items():
            path = project_file(args.model, name)
            if path.stat().st_size != item["bytes"] or file_sha(path) != sha_text(item["sha256"]):
                raise ValueError("Model asset changed: " + name)
        import numpy as np
        import torch
        from PIL import Image
        from fastrsi.object_stage_memory_v1 import encode

        torch.set_num_threads(4); torch.manual_seed(17)
        spec = importlib.util.spec_from_file_location(
            "object_stage_clef_source_v1", args.model / "joint_schema_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module; spec.loader.exec_module(module)
        tick = time.perf_counter()
        model, processor = module.load_release_model(
            args.model, device="cuda", attn_implementation="eager", local_files_only=True)
        torch.cuda.synchronize(); report["load_seconds"] = time.perf_counter() - tick
        report["model_revision"] = manifest["revision"]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        capture = {}
        hook = model.head.field_norm.register_forward_hook(
            lambda _module, _inputs, output: capture.update(field=output.detach().float().cpu()))
        features, upstream = [], {}
        report["status"] = "EXTRACTING"; save()
        try:
            for row in rows:
                tick = time.perf_counter(); images = []
                for ref in row["images"]:
                    path = project_file(args.project, ref["path"])
                    if file_sha(path) != ref["sha256"]:
                        raise ValueError("Image bytes changed")
                    with Image.open(path) as image:
                        if image.mode != "RGB" or hashlib.sha256(image.tobytes()).hexdigest() != ref["pixel_sha256"]:
                            raise ValueError("Image pixels changed")
                        images.append(image.copy())
                loaded_at = time.perf_counter()
                encoded = encode(module.encode_record, processor, {**row["request"], "images": images})
                if len(encoded.questions) != 1:
                    raise ValueError("Exactly one object-stage question required")
                encoded_at = time.perf_counter()
                batch = module.collate_records([encoded], processor.tokenizer.pad_token_id, "cuda")
                capture.clear()
                with torch.inference_mode():
                    logits = model(batch)[0][0]
                torch.cuda.synchronize(); inferred_at = time.perf_counter()
                field = capture.get("field")
                if field is None or field.shape != (1, 1024) or not torch.isfinite(field).all():
                    raise ValueError("Unexpected frozen field feature")
                options = encoded.questions[0].option_ids
                if logits.numel() != len(options) or not torch.isfinite(logits).all():
                    raise ValueError("Malformed upstream option logits")
                upstream[row["id"]] = dict(zip(options, logits.float().softmax(-1).cpu().tolist()))
                features.append(field[0].numpy())
                report["calls"].append(dict(
                    id=row["id"], group=row["group"], role=row["role"], subject=row["subject"],
                    image_count=4, image_read_seconds=loaded_at-tick,
                    encoding_seconds=encoded_at-loaded_at, inference_seconds=inferred_at-encoded_at,
                    total_seconds=inferred_at-tick, input_tokens=len(encoded.input_ids),
                    upstream_probabilities=upstream[row["id"]]))
                save()
        finally:
            hook.remove()
        matrix = np.stack(features)
        np.savez(args.output / "FEATURES.npz", features=matrix,
                 ids=np.asarray([row["id"] for row in rows]))
        vector = torch.nn.functional.normalize(torch.from_numpy(matrix).float(), dim=-1)
        versions, predictions = {}, {}
        report["status"] = "FITTING"; save()
        for name, roles in (("v1_seed", {"seed"}), ("v2_updated", {"seed", "update"})):
            indices = [i for i, row in enumerate(rows) if row["role"] in roles]
            selected = [labels[rows[i]["id"]] for i in indices]
            target = torch.tensor([label["eligible_to_skip"] for label in selected], dtype=torch.float32)
            weights = torch.tensor(grouped_class_weights(selected), dtype=torch.float32)
            torch.manual_seed(fit["seed"]); head = torch.nn.Linear(1024, 1)
            torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
            optimizer = torch.optim.AdamW(head.parameters(), lr=fit["learning_rate"],
                                          weight_decay=fit["weight_decay"])
            tick = time.perf_counter()
            for _ in range(fit["iterations"]):
                optimizer.zero_grad()
                logits = head(vector[indices])[:, 0]
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none") * weights).sum()
                loss.backward(); optimizer.step()
            training_seconds = time.perf_counter()-tick
            if not math.isfinite(float(loss.detach())):
                raise ValueError("Nonfinite external head loss")
            head.eval(); checkpoint = args.output / (name + "_head.pt")
            tick = time.perf_counter(); torch.save(head.state_dict(), checkpoint)
            checkpoint_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            with torch.no_grad():
                probability = head(vector)[:, 0].sigmoid().tolist()
            prediction_seconds = time.perf_counter()-tick
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probability):
                raise ValueError("Nonfinite external head prediction")
            predictions[name] = probability
            versions[name] = dict(
                parameters=1025, train_ids=[rows[i]["id"] for i in indices],
                train_groups=sorted({label["group"] for label in selected}),
                class_counts=dict(Counter(str(label["eligible_to_skip"]) for label in selected)),
                both_classes_supported=len({label["eligible_to_skip"] for label in selected}) == 2,
                sample_weights=weights.tolist(), fit_seconds=training_seconds,
                checkpoint_seconds=checkpoint_seconds, final_loss=float(loss.detach()),
                head_prediction_seconds=prediction_seconds,
                checkpoint_file=checkpoint.name, checkpoint_sha256=file_sha(checkpoint),
                fit_rule="Same zero initialization; seed vs seed+update refit. Frozen Clef.")
        (args.output / "TRAINING_BINDINGS.json").write_text(json.dumps(versions, indent=2, allow_nan=False))
        probe_predictions = [dict(
            id=row["id"], group=row["group"], subject=row["subject"],
            probabilities={name: values[i] for name, values in predictions.items()},
            eligible_predictions={name: values[i] >= protocol["threshold"] for name, values in predictions.items()},
            upstream_probabilities=upstream[row["id"]])
            for i, row in enumerate(rows) if row["role"] == "probe"]
        prediction_path = args.output / "PROBE_PREDICTIONS.json"
        prediction_path.write_text(json.dumps(probe_predictions, indent=2, allow_nan=False))
        report.update(status="PREDICTIONS_SAVED", versions=versions,
                      feature_sha256=file_sha(args.output / "FEATURES.npz"),
                      predictions_sha256=file_sha(prediction_path),
                      probe_labels_read=False, prediction_save_unix_ns=time.time_ns())
        save()
        # The probe file is neither parsed nor hashed before the saved versions/predictions.
        probe_path = args.inputs / "PROBE_SUPERVISION.json"
        probe_labels = validate_labels(read_json(probe_path), indexed, {"probe"})
        report["probe_labels_read"] = True
        report["probe_supervision_sha256"] = file_sha(probe_path)
        metrics = {}
        for version in versions:
            scored = []
            for prediction in probe_predictions:
                label = probe_labels[prediction["id"]]
                accepted = prediction["eligible_predictions"][version]
                scored.append(dict(id=label["id"], group=label["group"], subject=label["subject"],
                                   eligible_to_skip=label["eligible_to_skip"], accepted=accepted,
                                   false_accept=accepted and not label["eligible_to_skip"],
                                   missed_eligible=label["eligible_to_skip"] and not accepted))
            metrics[version] = dict(samples=len(scored),
                                    episode_groups=len({item["group"] for item in scored}),
                                    proposed_skip=sum(item["accepted"] for item in scored),
                                    false_accepts=sum(item["false_accept"] for item in scored),
                                    missed_eligible=sum(item["missed_eligible"] for item in scored), rows=scored)
        report.update(status="COMPLETED", metrics=metrics, threshold=protocol["threshold"],
                      deployment_approved=False, learning_net_savings_established=False,
                      label_scope=LABEL_SCOPE,
                      scope="Frozen Clef visual inference and grouped external-head fitting only. "
                            "Native subgoal/release fact supervision, not counterfactual route labels, "
                            "online control, calibrated success probabilities, cyclic distillation or net gain.")
        save()
        return report
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc()); save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/project"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "metrics": result.get("metrics"),
                      "full_wall_seconds": result["full_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
