"""Opt-in developmental lift/approach skill, using public RGB and robot8.

This fixed controller is an operation candidate, not a learned selector. A
commanded opening is an event, never a physical release certificate.
"""
import copy
import hashlib
import json
import re
import threading
import time

import numpy as np

from .planar_binding import intersect_plane

KEY = "local_bowl_transition_v1"
GOAL = "stack the right bowl on the left bowl and place them in the tray"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_recipe(value):
    expected = dict(version=KEY, max_local_actions=40, chunk_steps=5,
                    hover_z=.62, position_tolerance=.01, max_normalized_translation=.4,
                    open_finger_min=.03, search_radius_pixels=24,
                    minimum_match_score=.8, minimum_peak_gap=.05)
    if value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("Use the explicit first developmental recipe")
    return copy.deepcopy(value)


class BowlTransition:
    def __init__(self, *, session_id, enabled, source_rgb, source_rois, calibration, recipe, emit=None, eligibility_gate=None):
        from .native_goal_component_v1 import bind_initial_rgb
        if not session_id or type(enabled) is not bool:
            raise ValueError("Explicit root and opt-in required")
        self.session_id, self.enabled, self.source_rgb = session_id, enabled, source_rgb
        self.source_rois, self.calibration = source_rois, copy.deepcopy(calibration)
        self.recipe = validate_recipe(recipe)
        from .public_translation_gate_v1 import PublicTranslationGate, KEY as GATE
        if eligibility_gate not in (None, GATE) or (eligibility_gate is not None and not enabled):
            raise ValueError("Only the explicit opt-in translation-history gate")
        self.gate = None if eligibility_gate is None else PublicTranslationGate(self.recipe)
        self.recipe_sha256 = sha(self.recipe)
        self.bind_initial_rgb = bind_initial_rgb
        self.emit, self.events = emit, []
        self.initial = self.roi = self.template = None
        self.previous_sign = self.trigger_step = None
        self.phase, self.retired = "waiting", False
        self.entry_xy = None
        self.local_actions = self.local_requests = self.actual_vla_calls = 0
        self.seconds = 0.
        self.entered = False

    def event(self, stage, **fields):
        row = {"event":KEY, "stage":stage, "session_id":self.session_id, **fields}
        self.events.append(row)
        if self.emit is not None:
            self.emit(row)

    def submitted(self, chunk, before, after, status):
        actions = np.asarray(chunk["actions"])
        count = after-before
        if type(count) is not int or not 0 <= count <= len(actions):
            raise ValueError("Actual executed prefix required")
        if status.get("accepted") is False:
            return
        local = KEY in chunk.get("provenance", {})
        if local:
            self.local_actions += count
        else:
            for i, action in enumerate(actions[:count]):
                sign = float(action[6]) > 0
                if (self.gate is not None and not self.gate.closed and self.trigger_step is not None
                        and sign and self.phase == "waiting"):
                    self.gate.close()
                    self.retired = True
                    self.event("handoff", step=before+i+1, reason="PRE_INTERVENTION_WATCH_ENDED_BY_CLOSE_COMMAND",
                               physical_grasp_confirmed=False)
                if self.trigger_step is None and self.previous_sign is True and not sign:
                    self.trigger_step = before+i+1
                    self.event("opening_command", step=self.trigger_step, physical_release_confirmed=False)
                self.previous_sign = sign

    def track(self, rgb):
        import cv2
        x0, y0, x1, y1 = self.roi
        radius = self.recipe["search_radius_pixels"]
        left, top = max(0, x0-radius), max(0, y0-radius)
        right, bottom = min(rgb.shape[1], x1+radius), min(rgb.shape[0], y1+radius)
        area = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        scores = cv2.matchTemplate(area, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        h, w = self.template.shape
        work = scores.copy()
        work[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1] = -np.inf
        competitor = float(np.max(work))
        if (not np.isfinite(score) or score < self.recipe["minimum_match_score"]
                or (np.isfinite(competitor) and score-competitor < self.recipe["minimum_peak_gap"])):
            return None, dict(score=float(score), competitor=competitor if np.isfinite(competitor) else None, status="UNKNOWN")
        box = [left+x, top+y, left+x+w, top+y+h]
        center = [(box[0]+box[2]-1)/2, (box[1]+box[3]-1)/2]
        xyz = intersect_plane(center, self.calibration)
        return xyz, dict(score=float(score), competitor=competitor if np.isfinite(competitor) else None,
                         box_xyxy=box, world_plane_estimate=xyz.tolist(), status="APPEARANCE_MATCH_NOT_SEMANTIC_CERTIFICATE")

    def plan(self, observation, context):
        if context.get("session_id") != self.session_id:
            return None
        if context.get("task_description") != GOAL:
            raise ValueError("Keep the complete original goal")
        execution = context["execution"]
        if execution["max_steps"] != 400 or execution["replan_every_steps"] != 10:
            raise ValueError("Keep native 400/H10")
        match = re.fullmatch(r"libero_obs_(\d+)", observation["observation_id"])
        if match is None or context.get("source_observation_id") != observation["observation_id"]:
            raise ValueError("Current physical observation binding required")
        step = int(match[1])
        if not self.enabled or self.retired:
            return None
        rgb = np.ascontiguousarray(observation["sensors"]["front_rgb"]["data"])
        state = np.asarray(observation["sensors"]["proprio"]["data"])
        if rgb.shape != (256,256,3) or rgb.dtype != np.uint8 or state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Original finite public RGB/robot8 required")
        if self.initial is None:
            if step != 0:
                raise ValueError("Bind appearance only at initial observation")
            self.initial = rgb.copy()
            _, binding = self.bind_initial_rgb(self.source_rgb, rgb, self.source_rois, self.calibration)
            self.roi = binding["selected_boxes"][1]
            x0,y0,x1,y1 = self.roi
            import cv2
            self.template = cv2.cvtColor(rgb[y0:y1,x0:x1], cv2.COLOR_RGB2GRAY)
            self.event("initial_target_binding", step=step, binding=binding, roi=self.roi, online_sample_identity_used=False)
            return None
        if self.trigger_step is None:
            return None
        if self.local_actions >= self.recipe["max_local_actions"]:
            self.retired = True
            self.event("handoff", step=step, reason="LOCAL_ACTION_CAP", local_actions=self.local_actions)
            return None
        opened = state[6] >= self.recipe["open_finger_min"] and state[7] <= -self.recipe["open_finger_min"]
        if not opened:
            if self.phase != "waiting":
                self.retired = True
                self.event("handoff", step=step, reason="OPEN_FINGER_CONDITION_LOST")
            return None
        if self.gate is not None and self.phase == "waiting":
            evidence = self.gate.observe(step,state)
            self.event("eligibility_gate", **evidence)
            if not evidence["eligible"]:
                return None
        xyz, tracking = self.track(rgb)
        if xyz is None:
            if self.phase != "waiting":
                self.retired = True
            self.event("no_local_action", step=step, reason="APPEARANCE_UNKNOWN", tracking=tracking)
            return None
        if self.phase == "waiting":
            self.phase = "lift"
            self.entry_xy = state[:2].copy()
            self.event("enter", step=step, public_robot8=state.tolist(), tracking=tracking,
                       physical_release_confirmed=False, learned=False)
        if self.phase == "lift" and state[2] >= self.recipe["hover_z"]-self.recipe["position_tolerance"]:
            self.phase = "approach"
        # robot8 is float32. Build the protocol goal in float64 so numpy does
        # not round the fixed recipe height before the exact target check.
        goal = np.array([*(self.entry_xy if self.phase == "lift" else xyz[:2]),
                         self.recipe["hover_z"]], dtype=np.float64)
        if self.phase == "approach" and np.linalg.norm(state[:3]-goal) <= self.recipe["position_tolerance"]:
            self.retired = True
            self.event("handoff", step=step, reason="PUBLIC_EEF_APPROACH_TARGET_REACHED", public_robot8=state.tolist(), tracking=tracking)
            return None
        count = min(self.recipe["chunk_steps"], 400-step, self.recipe["max_local_actions"]-self.local_actions)
        tag = dict(version=KEY, session_id=self.session_id, observation_id=observation["observation_id"],
                   start_step=step, count=count, phase=self.phase, goal_xyz=goal.tolist(),
                   rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), recipe_sha256=self.recipe_sha256)
        self.event("planned_goal", **tag, tracking=tracking, public_robot8=state.tolist())
        return tag

    def install(self, *, policy_class=None):
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter as Adapter
        from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper as Policy
        from PhyAgentOS.runtime.skillruntime.policy.openpi import OpenPISkillRuntime as Runtime
        if policy_class is not None:
            Policy = policy_class
        original_input, original_output, original_infer, original_loop = Adapter.to_policy_input, Adapter.from_policy_output, Policy.infer, Runtime.run_policy_loop
        owner, local = self, threading.local()

        def input(adapter, observation, context):
            if getattr(local, "pending", None) is not None:
                raise ValueError("Previous inference not consumed")
            payload = original_input(adapter, observation, context)
            started = time.perf_counter()
            try:
                tag = owner.plan(observation, context)
            except ValueError as error:
                # Appearance/calibration uncertainty falls back before motion.
                owner.retired = True
                owner.event("handoff", reason="INPUT_OR_BINDING_REJECTED", error=str(error))
                tag = None
            finally:
                owner.seconds += time.perf_counter()-started
            local.pending = dict(payload=payload, tag=tag, root=context.get("session_id") == owner.session_id)
            return payload

        def infer(policy, payload):
            pending, local.pending = getattr(local,"pending",None), None
            if pending is None or pending["payload"] is not payload:
                raise ValueError("Local dispatch must consume its exact adapter payload")
            if pending["tag"] is None:
                if pending["root"]:
                    owner.actual_vla_calls += 1
                return original_infer(policy, payload)
            tag = pending["tag"]
            owner.local_requests += 1
            return dict(actions=np.zeros((tag["count"],7),np.float32),
                        policy_meta={KEY:tag, "prediction_origin":"public_feedback_skill_request_not_vla"})

        def output(adapter, answer, context):
            chunk = original_output(adapter, answer, context)
            if KEY in answer.get("policy_meta", {}):
                chunk.setdefault("provenance",{})[KEY] = answer["policy_meta"][KEY]
            return chunk

        def loop(runtime, skill_ctx, handle, adapter_plan, policy_client):
            if skill_ctx.session.session_id != owner.session_id:
                return original_loop(runtime, skill_ctx, handle, adapter_plan, policy_client)
            if owner.entered:
                raise ValueError("Do not restart the root local controller")
            owner.entered = True
            class SameHandle:
                def action_chunk(self, chunk):
                    before = handle.session_state.step_index
                    result = handle.action_chunk(chunk)
                    owner.submitted(chunk, before, handle.session_state.step_index, result)
                    return result
                def __getattr__(self, name):
                    return getattr(handle, name)
            return original_loop(runtime, skill_ctx, SameHandle(), adapter_plan, policy_client)

        Adapter.to_policy_input, Adapter.from_policy_output, Policy.infer, Runtime.run_policy_loop = input, output, infer, loop
        def restore():
            Adapter.to_policy_input, Adapter.from_policy_output, Policy.infer, Runtime.run_policy_loop = original_input, original_output, original_infer, original_loop
        return restore


class PublicFeedbackTarget:
    def __init__(self, config):
        self.root_session = config["root_session"]
        self.goal = config["task_description"]
        self.recipe = validate_recipe(config["local_transition_recipe"])
        self.used = 0

    def execute(self, runtime, chunk, original):
        tag = chunk.get("provenance", {}).get(KEY)
        if tag is None:
            return original(runtime, chunk), None
        allowed = {"version","session_id","observation_id","start_step","count","phase","goal_xyz","rgb_sha256","recipe_sha256"}
        count = min(5, 400-runtime.step_idx, 40-self.used)
        pixels = np.ascontiguousarray(runtime._last_obs["agentview_image"])
        if (set(tag) != allowed or tag["version"] != KEY or runtime.session_id != self.root_session
                or tag["session_id"] != runtime.session_id or tag["start_step"] != runtime.step_idx
                or tag["observation_id"] != "libero_obs_%d" % runtime.step_idx
                or runtime.language != GOAL or self.goal != GOAL or runtime.config["max_steps"] != 400
                or type(tag["count"]) is not int or tag["count"] != count or count <= 0
                or tag["phase"] not in ("lift","approach") or tag["recipe_sha256"] != sha(self.recipe)
                or tag["rgb_sha256"] != hashlib.sha256(pixels.tobytes()).hexdigest()
                or np.asarray(chunk["actions"]).shape != (count,7) or np.any(np.asarray(chunk["actions"]) != 0)):
            raise ValueError("Local skill request differs from current root/public input/budget")
        goal = np.asarray(tag["goal_xyz"],dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all() or goal[2] != .62 or np.any(np.abs(goal[:2]) > .5):
            raise ValueError("Local target outside declared hover workspace")
        controller = runtime.env.env.robots[0].controller
        if (not controller.use_delta or not controller.use_ori or controller.impedance_mode != "fixed"
                or not np.array_equal(controller.input_min, [-1.]*6) or not np.array_equal(controller.input_max,[1.]*6)
                or not np.allclose(controller.output_min,[-.05]*3+[-.5]*3,atol=0,rtol=0)
                or not np.allclose(controller.output_max,[.05]*3+[.5]*3,atol=0,rtol=0)):
            raise ValueError("Unverified original controller scaling")
        start, index = runtime.step_idx, len(runtime._episode_chunks)
        actions, public_feedback, reward = [], [], 0.
        tick = time.perf_counter()
        for _ in range(count):
            observed = runtime._last_obs
            position = np.asarray(observed["robot0_eef_pos"],dtype=float)
            qpos = np.asarray(observed["robot0_gripper_qpos"],dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(qpos).all():
                raise ValueError("Public position/finger observation invalid")
            # Each physical step uses its newly returned public position.
            command = np.clip((goal-position)/.05, -.4, .4)
            action = np.r_[command,0.,0.,0.,-1.].astype(np.float32)
            public_feedback.append(dict(step=runtime.step_idx, position=position.tolist(), goal_xyz=goal.tolist()))
            actions.append(action.tolist())
            result = original(runtime,{**chunk,"actions":[action.tolist()]})
            reward += float(result.get("reward",0.))
            self.used += 1
            if runtime.done:
                break
        substeps = runtime._episode_chunks[index:]
        if len(substeps) != len(actions):
            raise ValueError("Actual local execution records incomplete")
        runtime._episode_chunks[index:] = [{**substeps[-1], "first_step":start+1,
            "executed_steps":runtime.step_idx-start, "requested_steps":count,
            "reward":reward, "action_shape":[count,7]}]
        runtime._last_status.update(reward=reward,episode_summary=runtime._episode_summary())
        return dict(runtime._last_status), dict(operation_kind=KEY, start_step=start,end_step=runtime.step_idx,
            actual_actions=actions,public_feedback=public_feedback,candidate_sha256=sha(self.recipe),
            object_pose_used=False,physical_release="UNKNOWN",seconds=time.perf_counter()-tick,
            request=copy.deepcopy(tag))
