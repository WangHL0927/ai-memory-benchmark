"""金标、政策校验和拆分评分共用的统计函数。"""
from collections import defaultdict
import json
from pathlib import Path

from tools.paths import ROOT


def require(ok, message):
    if not ok:
        raise ValueError(message)


def policy_ref(policy):
    return {k: policy[k] for k in ("id", "version")}


def load_policy(path=None, reference=None):
    """默认按金标声明定位；标识只允许文件名字符，避免拼接越出 policies。"""
    if path is None:
        require(isinstance(reference, dict) and all(isinstance(reference.get(k), str)
                and reference[k] and all(c.isalnum() or c in "_-." for c in reference[k])
                for k in ("id", "version")), "gold 缺少有效 policy id/version")
        path = ROOT / "policies" / (reference["id"] + "-" + reference["version"] + ".json")
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_policy(policy)
    return policy


def validate_policy(policy):
    require(isinstance(policy, dict), "政策必须为对象")
    for key in ("id", "version"):
        require(isinstance(policy.get(key), str) and bool(policy[key]), "政策缺 " + key)
    for key in ("categories", "reject_reasons", "condition_types", "scope_values", "flags"):
        values = policy.get(key, [] if key == "flags" else None)
        require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
                and len(values) == len(set(values)), "政策枚举无效: " + key)
    require(bool(policy["categories"]) and bool(policy["scope_values"]), "政策类别/范围不能为空")
    require(policy.get("superseded") in ("final_state_only", "keep_history"), "政策 superseded 无效")
    require(isinstance(policy.get("attributes"), dict), "政策 attributes 无效")
    for name, spec in policy["attributes"].items():
        require(isinstance(spec, dict) and isinstance(spec.get("values"), list)
                and bool(spec["values"]) and all(isinstance(v, str) and v for v in spec["values"]), "政策属性无效: " + name)
        allowed = spec.get("allowed_by_category", {})
        require(isinstance(allowed, dict) and all(c in policy["categories"] and isinstance(v, list)
                and all(x in spec["values"] for x in v) for c, v in allowed.items()), "政策属性类别约束无效: " + name)


def attribute_valid(policy, name, category, value):
    spec = policy["attributes"][name]
    return value in spec["values"] and value in spec.get("allowed_by_category", {}).get(category, spec["values"])


def scope_error(value, policy, projects):
    if not isinstance(value, str) or not any(value == v or (v == "project:*" and value.startswith("project:")
            and len(value) > len("project:")) for v in policy["scope_values"]):
        return "scope 不符合政策 scope_values"
    if value.startswith("project:") and value[len("project:"):] not in projects:
        return "scope 的项目不在 gold.project 列表中: " + value[len("project:"):]
    return ""


def reject_valid(reason, policy):
    return reason in policy["reject_reasons"] and reason != "prior_context" and not (
        reason == "superseded" and policy["superseded"] == "keep_history")


def validate_gold(gold, policy=None):
    policy = policy or load_policy(reference=gold.get("policy"))
    validate_policy(policy)
    require(gold.get("policy") == policy_ref(policy), "gold 与政策 id/version 不一致")
    require(isinstance(gold.get("project"), list) and bool(gold["project"])
            and all(isinstance(p, str) and p.strip() for p in gold["project"]), "gold.project 必须为非空项目列表")
    for field in ("sample_id", "session_id", "turns", "layer", "standard_version", "frozen_at", "U", "N"):
        require(field in gold, f"gold 缺字段 {field}")
    require(bool(gold["frozen_at"]), "gold 尚未冻结")
    require(gold["layer"] in ("random", "targeted"), "会话 layer 无效")
    require(isinstance(gold["turns"], int) and gold["turns"] >= 0, "turns 无效")
    require(isinstance(gold["U"], list) and isinstance(gold["N"], list), "U/N 必须为列表")
    if "frozen_policy" in gold:
        require(gold["frozen_policy"] == gold["policy"], "冻结的政策声明已被修改")
    if "frozen_project" in gold:
        require(gold["frozen_project"] == gold["project"], "冻结的 project 已被修改")
    snapshot = gold.get("blind_snapshot")
    if snapshot is not None:
        require(snapshot == {"U": [u for u in gold["U"] if u["origin"] == "blind"], "N": gold["N"]},
                "冻结的盲标快照与当前 U/N 不一致")
    ids = set()
    for kind in ("U", "N"):
        for unit in gold[kind]:
            for field in ("unit_id", "statement", "conditions", "scope", "evidence", "uncertain", "origin", "note"):
                require(field in unit, f"{kind} 单元缺字段 {field}")
            uid = unit["unit_id"]
            require(isinstance(uid, str) and uid and uid not in ids, "单元 id 为空或重复")
            ids.add(uid)
            require(type(unit["uncertain"]) is bool, f"{uid}: uncertain 必须为布尔值")
            require(unit["origin"] == "blind" or unit["origin"].startswith("added_by_run:"), f"{uid}: origin 无效")
            require(isinstance(unit["statement"], str) and bool(unit["statement"].strip()), f"{uid}: statement 为空")
            require(not scope_error(unit["scope"], policy, gold["project"]), f"{uid}: " + scope_error(unit["scope"], policy, gold["project"]))
            require(isinstance(unit["conditions"], list), f"{uid}: conditions 必须为列表")
            for condition in unit["conditions"]:
                require(condition.get("type") in policy["condition_types"] and bool(condition.get("text"))
                        and type(condition.get("key")) is bool, f"{uid}: 条件字段无效")
            require(isinstance(unit["evidence"], list) and bool(unit["evidence"]), f"{uid}: 缺少证据")
            for evidence in unit["evidence"]:
                require(type(evidence.get("turn")) is int and 1 <= evidence["turn"] <= gold["turns"]
                        and bool(evidence.get("quote")), f"{uid}: evidence 无效")
            if kind == "U":
                require(unit.get("category") in policy["categories"], f"{uid}: category 无效")
                require(isinstance(unit.get("attributes"), dict), f"{uid}: attributes 无效")
                for name in policy["attributes"]:
                    require(attribute_valid(policy, name, unit["category"], unit["attributes"].get(name)),
                            f"{uid}: attributes.{name} 无效或不适用于该类别")
            else:
                require(reject_valid(unit.get("reject_reason"), policy), f"{uid}: 负例原因无效或政策不允许")
                require(unit.get("layer") in ("random", "targeted"), f"{uid}: 缺少负例 layer")


def items(run):
    require(isinstance(run, dict) and isinstance(run.get("D"), list), "run 应为含 D 列表的对象")
    return run["D"]


def add_counts(values):
    result = defaultdict(float)
    for value in values:
        for k, v in value.items():
            result[k] += v
    return dict(result)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values)-1)*p
    lo = int(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)
