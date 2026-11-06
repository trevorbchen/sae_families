#!/usr/bin/env python3
"""Prefill residual-SAE GUI notes with local Kimi-VL suggestions."""

import argparse
import base64
import collections
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_TARGETS = (
    "direct_mse",
    "grouped_objective_laplacian",
    "grouped_objective_fourier_low",
    "grouped_objective_fourier_mid",
    "grouped_objective_fourier_high",
    "tandem_canonical",
)


PROMPT_VERSION = "frequency-family-v2"


PROMPT = """You are proposing an editable label for one causal autoencoder feature.
Each supplied image is one subject. Within each image, read left-to-right then
top-to-bottom: -10, -4, -2, 0, +2, +4, +10 feature-strength units. The 0 image
is untouched. Compare every subject only against its own 0 image and identify
changes that repeat across subjects. Extreme strengths may be artifacts.

The scientific goal is to find features enriched in particular SAE objective
families, especially genuinely frequency- or scale-selective features. Look for:
- low-frequency effects: global brightness or color cast, broad lighting
  gradients, background illumination, and face-wide shading;
- texture changes that preserve identity, pose, expression, and geometry;
- subject-edge effects versus background-edge effects;
- spatial scale, such as coarse hair mass versus individual hair strands;
- structured noise or repeated patterns, not generic corruption;
- selective edge/detail changes in eyes, teeth, hair, or face boundaries;
- fine detail isolated to one color channel;
- precise size and location: one fine shadow, several small shadows, a broad
  cheek shadow, a face-wide region, background only, and so on.

IMPORTANT: almost every strong perturbation has a slight softening/blur side
effect. Generic blur or softness is NOT a useful feature label and must never be
the primary effect. If slight blur is the only repeatable effect, return
primary_effect="unclear", effect_category="unclear", and
generic_blur_only=true. A true blur-kernel feature is only credible when its
scale and spatial support are specific, it dominates at moderate strengths,
and it is consistent across subjects.

Be skeptical when pose, age, expression, identity, or face shape also changes.
Such a feature may be semantic and merely correlated with frequency. The
cleanest frequency-selective candidates preserve identity and geometry while
changing only texture, edge strength, structured noise, color detail, or
spatial scale. Do not claim an effect merely because it appears in one subject.

Return JSON only with exactly these keys:
primary_effect (specific short phrase or "unclear"),
effect_category (lighting/color/texture/edge_strength/scale_structure/
structured_noise/channel_specific_detail/semantic/artifact/unclear),
candidate_type (frequency_selective/semantic/mixed/artifact/unclear),
frequency_band (low/mid/high/multiscale/not_frequency/unclear),
spatial_scale (global/broad/regional/small_local/fine/multiple_small/mixed/unclear),
spatial_distribution (global/face_wide/background/subject_interior/subject_edges/
background_edges/named_local_regions/multiple_disconnected_regions/mixed/unclear),
affected_regions (array of short, anatomically or spatially precise phrases),
direction (one short description of negative-to-positive change),
secondary_effects (array of short phrases),
semantic_confounds (array chosen from pose/age/expression/identity/face_shape or empty),
frequency_selective_candidate (true/false/uncertain),
texture_only (true/false/uncertain),
generic_blur_only (true/false),
subject_consistency (number 0 to 1),
identity_and_geometry_preserved (true/false/uncertain),
artifact_onset (signed strength or "none"),
confidence (number 0 to 1),
evidence (one concise sentence).
Use primary_effect="unclear" when the recurring effect is weak or visible only
at artifact-level strengths."""


STRICT_SUBJECT_PROMPT = """You are performing a neutral visual-difference audit.
The image contains one subject under seven interventions, ordered left-to-right
then top-to-bottom: -10, -4, -2, 0, +2, +4, +10. The 0 panel is the untouched
reference. This is one subject only; do not infer cross-subject consistency.

First compare -4, -2, +2, and +4 against 0. Describe only a repeatable,
approximately monotonic effect visible at those moderate strengths. Use -10 and
+10 only to report artifact onset. Explicitly inspect expression, pose, face
shape, apparent age, and identity before considering texture or frequency.
If eye aperture, mouth shape, head orientation, or facial geometry moves, use
effect_category="semantic", list the confound, and set
frequency_selective_candidate=false. Conversely, do not infer an expression
change from luminance alone: expression requires visible geometric movement.

Do not use the phrases \"texture changes\", \"softening\", or \"blur\" as a
primary effect. If that is all that is visible, label the feature unclear and
set generic_blur_only=true. Brightness, illumination, or color changes belong
to lighting/color, never texture. A texture label must name the texture and its
location or scale. Report direction explicitly as \"negative: ...; positive:
...\". Set confidence=1 only when the moderate-strength effect is unambiguous.

Return JSON only with exactly these keys:
primary_effect (specific short phrase or \"unclear\"),
effect_category (lighting/color/texture/edge_strength/scale_structure/
structured_noise/channel_specific_detail/semantic/artifact/unclear),
candidate_type (frequency_selective/semantic/mixed/artifact/unclear),
frequency_band (low/mid/high/multiscale/not_frequency/unclear),
spatial_scale (global/broad/regional/small_local/fine/multiple_small/mixed/unclear),
spatial_distribution (global/face_wide/background/subject_interior/subject_edges/
background_edges/named_local_regions/multiple_disconnected_regions/mixed/unclear),
affected_regions (array of precise phrases),
direction (negative-to-positive description in the required format),
secondary_effects (array of short phrases),
semantic_confounds (array chosen from pose/age/expression/identity/face_shape or empty),
frequency_selective_candidate (true/false/uncertain),
texture_only (true/false/uncertain),
generic_blur_only (true/false),
subject_consistency (1 only if monotonic within this sheet, otherwise 0),
identity_and_geometry_preserved (true/false/uncertain),
artifact_onset (signed strength or \"none\"),
confidence (number 0 to 1),
evidence (one concise sentence grounded in moderate strengths)."""


def request_json(url, payload=None, attempts=8):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30, 2 ** attempt))


def decode_data_url(value):
    encoded = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def subject_sheet(variants, cell=160):
    images = [decode_data_url(item["image"]) for item in variants]
    labels = [item["label"].replace(" scale", "") for item in variants]
    cols = 4
    rows = 2
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, cols)
        sheet.paste(image.resize((cell, cell)), (col * cell, row * (cell + 22)))
        draw.text((col * cell + 4, row * (cell + 22) + cell + 3), label, fill="black")
    return sheet


def most_common(values, default="unclear"):
    values = [str(value).strip() for value in values if str(value).strip()]
    return collections.Counter(values).most_common(1)[0][0] if values else default


def as_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or str(value).strip().lower() in {"", "none", "unclear"}:
        return []
    return [str(value).strip()]


def aggregate_subject_labels(labels):
    labels = [label for label in labels if isinstance(label, dict)]
    if not labels:
        raise ValueError("Kimi returned an empty label array")
    primary_counts = collections.Counter(
        str(label.get("primary_effect", "unclear")).strip().lower() for label in labels
    )
    primary_key, agreement_count = primary_counts.most_common(1)[0]
    agreement = agreement_count / len(labels)
    matching = [
        label for label in labels
        if str(label.get("primary_effect", "unclear")).strip().lower() == primary_key
    ]
    if agreement < 0.4:
        primary = "unclear"
        matching = labels
    else:
        primary = str(matching[0].get("primary_effect", primary_key)).strip()
    regions = list(dict.fromkeys(
        region for label in matching for region in as_list(label.get("affected_regions"))
    ))
    secondary = list(dict.fromkeys(
        effect for label in matching for effect in as_list(label.get("secondary_effects"))
    ))
    confidences = []
    for label in matching:
        try:
            confidences.append(float(label.get("confidence", 0)))
        except (TypeError, ValueError):
            pass
    return {
        "primary_effect": primary,
        "effect_category": most_common(
            [label.get("effect_category", "unclear") for label in matching]
        ),
        "candidate_type": most_common(
            [label.get("candidate_type", "unclear") for label in matching]
        ),
        "frequency_band": most_common(
            [label.get("frequency_band", "unclear") for label in matching]
        ),
        "spatial_scale": most_common(
            [label.get("spatial_scale", "unclear") for label in matching]
        ),
        "spatial_distribution": most_common(
            [label.get("spatial_distribution", "unclear") for label in matching]
        ),
        "affected_regions": regions,
        "direction": most_common([label.get("direction", "unclear") for label in matching]),
        "secondary_effects": secondary,
        "semantic_confounds": list(dict.fromkeys(
            effect for label in matching
            for effect in as_list(label.get("semantic_confounds"))
        )),
        "frequency_selective_candidate": most_common(
            [label.get("frequency_selective_candidate", "uncertain") for label in matching],
            "uncertain",
        ),
        "texture_only": most_common(
            [label.get("texture_only", "uncertain") for label in matching], "uncertain"
        ),
        "generic_blur_only": most_common(
            [label.get("generic_blur_only", False) for label in matching], False
        ),
        "subject_consistency": agreement,
        "identity_and_geometry_preserved": most_common(
            [label.get("identity_and_geometry_preserved", "uncertain") for label in matching],
            "uncertain",
        ),
        "artifact_onset": most_common(
            [label.get("artifact_onset", "none") for label in matching], "none"
        ),
        "confidence": (sum(confidences) / len(confidences) if confidences else 0) * agreement,
        "evidence": (
            f"Per-subject labels agreed on {primary} in "
            f"{agreement_count}/{len(labels)} panels."
        ),
    }


def bool_vote(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def strict_category(label):
    """Normalize obvious category contradictions before voting."""
    primary = str(label.get("primary_effect", "unclear")).strip().lower()
    category = str(label.get("effect_category", "unclear")).strip().lower()
    direction = str(label.get("direction", "")).strip().lower()
    secondary = " ".join(str(value).lower() for value in as_list(
        label.get("secondary_effects")
    ))
    regions = " ".join(str(value).lower() for value in as_list(
        label.get("affected_regions")
    ))
    text = " ".join((primary, direction, secondary, regions))
    if any(term in text for term in (
        "expression", "pose", "identity", "face shape", "face_shape", "age",
    )):
        return "semantic"
    if primary in {"texture changes", "softening", "blur", "generic blur"}:
        return "unclear"
    if category == "texture" and any(
        term in text for term in ("brightness", "illumination", "lighting", "color")
    ):
        return "lighting/color"
    return category


def inferred_semantic_confounds(label):
    fields = [
        label.get("primary_effect", ""),
        label.get("direction", ""),
        *as_list(label.get("affected_regions")),
        *as_list(label.get("secondary_effects")),
        *as_list(label.get("semantic_confounds")),
    ]
    text = " ".join(str(value).lower() for value in fields)
    aliases = {
        "pose": ("pose", "head turn", "head tilt"),
        "age": ("age", "older", "younger"),
        "expression": ("expression", "smile", "mouth", "eyes open", "eyes closed"),
        "identity": ("identity",),
        "face_shape": ("face shape", "face_shape", "jaw shape", "head shape"),
    }
    return [name for name, terms in aliases.items() if any(term in text for term in terms)]


def aggregate_subject_labels_strict(labels):
    """Aggregate independent subject calls with conservative agreement gates."""
    labels = [label for label in labels if isinstance(label, dict)]
    if not labels:
        raise ValueError("Kimi returned no parseable per-subject labels")
    categories = [strict_category(label) for label in labels]
    category, category_count = collections.Counter(categories).most_common(1)[0]
    agreement = category_count / len(labels)
    matching = [
        label for label, value in zip(labels, categories) if value == category
    ]
    if agreement < 0.6:
        category = "unclear"
        matching = labels

    primary_values = [
        str(label.get("primary_effect", "unclear")).strip()
        for label in matching
    ]
    primary = most_common(primary_values, "unclear")
    if category == "unclear":
        primary = "unclear"

    regions = list(dict.fromkeys(
        region for label in matching for region in as_list(label.get("affected_regions"))
    ))
    secondary = list(dict.fromkeys(
        effect for label in matching for effect in as_list(label.get("secondary_effects"))
    ))
    confounds = list(dict.fromkeys(
        effect for label in labels for effect in inferred_semantic_confounds(label)
    ))
    confidences = []
    for label in matching:
        try:
            confidences.append(float(label.get("confidence", 0)))
        except (TypeError, ValueError):
            pass

    candidate_votes = [
        bool_vote(label.get("frequency_selective_candidate")) for label in labels
    ]
    candidate_true = sum(vote is True for vote in candidate_votes)
    identity_votes = [
        bool_vote(label.get("identity_and_geometry_preserved")) for label in labels
    ]
    identity_true = sum(vote is True for vote in identity_votes)
    required = max(1, (3 * len(labels) + 4) // 5)
    clean_frequency = (
        agreement >= 0.6
        and candidate_true >= required
        and identity_true >= required
        and not confounds
        and category in {
            "lighting/color", "texture", "edge_strength", "scale_structure",
            "structured_noise", "channel_specific_detail",
        }
    )
    unclear = category == "unclear"

    return {
        "primary_effect": primary,
        "effect_category": category,
        "candidate_type": "frequency_selective" if clean_frequency else (
            "unclear" if unclear else "mixed"
        ),
        "frequency_band": "unclear" if unclear else most_common(
            [label.get("frequency_band", "unclear") for label in matching]
        ),
        "spatial_scale": most_common(
            [label.get("spatial_scale", "unclear") for label in matching]
        ),
        "spatial_distribution": most_common(
            [label.get("spatial_distribution", "unclear") for label in matching]
        ),
        "affected_regions": regions,
        "direction": "unclear" if unclear else most_common(
            [label.get("direction", "unclear") for label in matching]
        ),
        "secondary_effects": secondary,
        "semantic_confounds": confounds,
        "frequency_selective_candidate": clean_frequency,
        "texture_only": "uncertain" if unclear else most_common(
            [label.get("texture_only", "uncertain") for label in matching], "uncertain"
        ),
        "generic_blur_only": all(
            bool_vote(label.get("generic_blur_only")) is True for label in matching
        ),
        "subject_consistency": agreement,
        "identity_and_geometry_preserved": identity_true >= required,
        "artifact_onset": most_common(
            [label.get("artifact_onset", "none") for label in matching], "none"
        ),
        "confidence": min(
            0.25 if unclear else 0.95,
            (sum(confidences) / len(confidences) if confidences else 0) * agreement,
        ),
        "evidence": (
            f"Effect category agreed in {category_count}/{len(labels)} independent "
            "subject panels; frequency candidacy required agreement and no semantic confound."
        ),
    }


def generate_response(model, processor, sheets, prompt, max_new_tokens):
    content = [{"type": "image", "image": image} for image in sheets]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = processor(images=sheets, text=text, return_tensors="pt", padding=True).to(
        model.device
    )
    import torch
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [tokens[len(prompt_ids):] for prompt_ids, tokens in zip(inputs.input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def retry_keys_from(root):
    keys = set()
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = record.get("label")
            if not isinstance(label, dict):
                continue
            primary = str(label.get("primary_effect", "unclear")).strip().lower()
            if primary != "unclear":
                keys.add((
                    record.get("source", ""),
                    record.get("requested_component", ""),
                    int(record.get("rank", -1)),
                ))
    return keys


def parse_json(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1].removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return aggregate_subject_labels(parsed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    recovered = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            recovered.append(value)
        cursor = start + length
    if recovered:
        return aggregate_subject_labels(recovered)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Kimi response contains no JSON object")
    return json.loads(text[start:end + 1])


def concise_note(label):
    primary = str(label.get("primary_effect", "unclear")).strip()
    direction = str(label.get("direction", "")).strip()
    regions = ", ".join(as_list(label.get("affected_regions")))
    confidence = label.get("confidence", 0)
    pieces = [f"Kimi suggestion: {primary}"]
    candidate_type = str(label.get("candidate_type", "")).strip()
    frequency = str(label.get("frequency_band", "")).strip()
    scale = str(label.get("spatial_scale", "")).strip()
    distribution = str(label.get("spatial_distribution", "")).strip()
    if candidate_type and candidate_type != "unclear":
        pieces.append(f"Type: {candidate_type}")
    if frequency and frequency != "unclear":
        pieces.append(f"Band: {frequency}")
    if scale and scale != "unclear":
        pieces.append(f"Scale: {scale}")
    if distribution and distribution != "unclear":
        pieces.append(f"Support: {distribution}")
    if direction and direction.lower() != "unclear":
        pieces.append(f"Direction: {direction}")
    if regions:
        pieces.append(f"Regions: {regions}")
    try:
        pieces.append(f"Confidence: {float(confidence):.2f}")
    except (TypeError, ValueError):
        pass
    return ". ".join(piece.rstrip(".") for piece in pieces) + "."


def annotation_payload(feature, comment):
    annotation = feature.get("annotation") or {}
    return {
        "key": feature["annotation_key"],
        "source": feature["source"],
        "source_label": feature["source_label"],
        "rank": feature["rank"],
        "feature": feature["feature"],
        "interesting": bool(annotation.get("interesting", False)),
        "comment": comment,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=520)
    parser.add_argument("--per-subject", action="store_true")
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument(
        "--replace-kimi-notes", action="store_true",
        help="replace prior Kimi suggestion comments while preserving human comments",
    )
    parser.add_argument(
        "--targets", nargs="+", default=DEFAULT_TARGETS,
        help="source or source:component entries",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                record = json.loads(line)
                if record.get("note_posted") or record.get("skipped_existing"):
                    completed.add((
                        record["source"], record.get("requested_component", ""),
                        int(record["rank"]),
                    ))
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    endpoint = args.endpoint.rstrip("/")
    retry_keys = retry_keys_from(args.retry_from) if args.retry_from else None

    for target in args.targets:
        source, separator, component = target.partition(":")
        if not separator:
            component = ""
        for rank in range(args.start, args.end):
            if retry_keys is not None and (source, component, rank) not in retry_keys:
                continue
            if (source, component, rank) in completed:
                continue
            query_values = {"source": source, "rank": rank}
            if component:
                query_values["component"] = component
            query = urllib.parse.urlencode(query_values)
            feature = request_json(f"{endpoint}/api/feature?{query}")
            annotation = feature.get("annotation") or {}
            existing_comment = str(annotation.get("comment", "")).strip()
            already_interesting = bool(annotation.get("interesting", False))
            replaceable = (
                args.replace_kimi_notes
                and existing_comment.startswith("Kimi suggestion:")
            )
            if already_interesting or (existing_comment and not replaceable):
                record = {
                    "source": source,
                    "requested_component": component,
                    "rank": rank,
                    "feature": feature["feature"],
                    "interesting": bool(annotation.get("interesting", False)),
                    "dataset_active_fraction": feature.get("dataset_active_fraction"),
                    "active_rms": feature.get("active_rms"),
                    "skipped_interesting": already_interesting,
                    "skipped_existing": True,
                    "note_posted": False,
                }
            else:
                sheets = [
                    subject_sheet(item["variants"])
                    for item in feature["images"][:args.examples]
                ]
                subject_labels = []
                parse_errors = []
                responses = []
                request_sheets = [[sheet] for sheet in sheets] if args.per_subject else [sheets]
                request_prompt = STRICT_SUBJECT_PROMPT if args.per_subject else PROMPT
                for request_images in request_sheets:
                    response = generate_response(
                        model, processor, request_images, request_prompt, args.max_new_tokens
                    )
                    responses.append(response)
                    try:
                        subject_labels.append(parse_json(response))
                    except Exception as error:
                        parse_errors.append(str(error))
                try:
                    label = (
                        aggregate_subject_labels_strict(subject_labels)
                        if args.per_subject else subject_labels[0]
                    )
                    parse_error = "; ".join(parse_errors) or None
                except Exception as error:
                    label = {"primary_effect": "unparseable", "raw_response": responses}
                    parse_error = str(error)
                note = concise_note(label)

                # Re-read before posting so a note typed during inference wins.
                latest = request_json(f"{endpoint}/api/feature?{query}")
                latest_annotation = latest.get("annotation") or {}
                latest_comment = str(latest_annotation.get("comment", "")).strip()
                latest_interesting = bool(latest_annotation.get("interesting", False))
                latest_replaceable = (
                    args.replace_kimi_notes
                    and latest_comment.startswith("Kimi suggestion:")
                )
                if latest_interesting or (latest_comment and not latest_replaceable):
                    posted = False
                    skipped_race = True
                else:
                    request_json(
                        f"{endpoint}/api/annotation",
                        annotation_payload(latest, note),
                    )
                    posted = True
                    skipped_race = False
                record = {
                    "source": source,
                    "requested_component": component,
                    "source_label": feature["source_label"],
                    "component_label": feature.get("component_label", ""),
                    "rank": rank,
                    "feature": feature["feature"],
                    "annotation_key": feature["annotation_key"],
                    "interesting": bool(annotation.get("interesting", False)),
                    "dataset_active_fraction": feature.get("dataset_active_fraction"),
                    "active_rms": feature.get("active_rms"),
                    "prompt_version": (
                        f"{PROMPT_VERSION}-neutral-per-subject-v3" if args.per_subject
                        else PROMPT_VERSION
                    ),
                    "label": label,
                    "subject_labels": subject_labels if args.per_subject else None,
                    "parse_error": parse_error,
                    "suggested_note": note,
                    "note_posted": posted,
                    "skipped_existing": False,
                    "skipped_user_race": skipped_race,
                }
            with out.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({
                "source": source,
                "rank": rank,
                "feature": feature["feature"],
                "posted": record.get("note_posted", False),
                "label": (record.get("label") or {}).get("primary_effect"),
            }), flush=True)


if __name__ == "__main__":
    main()
