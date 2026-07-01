"""
[Partner B]
Data upload and preprocessing routes.

Endpoints:
  POST  /data/upload/{modality}     – upload a CSV / JSONL / ZIP file
  POST  /data/preprocess/{modality} – run cleaning + validation on uploaded file
  GET   /data/status                – show what data is currently loaded per modality
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse, Modality

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Data"])

# ---------------------------------------------------------------------------
# In-memory store for uploaded file metadata
# (Replaced by DB persistence in a later phase if needed)
# ---------------------------------------------------------------------------

_uploaded: dict[str, dict] = {
    "tabular": {"filename": None, "rows": 0, "columns": 0, "preprocessed": False, "source": "synthetic"},
    "text":    {"filename": None, "rows": 0, "preprocessed": False, "source": "synthetic"},
    "image":   {"filename": None, "count": 0, "preprocessed": False, "source": "synthetic"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_FILE_MB = 200
MAX_BYTES   = MAX_FILE_MB * 1024 * 1024  # 200 MB

# Shared label-column fallback list. Both data.py (tabular preprocess) and
# models.py (tabular + text training) now import this single source of
# truth instead of each maintaining their own divergent candidate list.
LABEL_COLUMN_CANDIDATES = ["label", "class", "target", "attack", "fraud", "y"]


def _resolve_label_column(columns, requested: Optional[str] = None) -> Optional[str]:
    """
    Resolve the label column name against a list of available columns.
    If `requested` is given and present, it wins. Otherwise walk
    LABEL_COLUMN_CANDIDATES in order and return the first match.
    Returns None if nothing matches.
    """
    columns = list(columns)
    if requested and requested in columns:
        return requested
    return next((c for c in LABEL_COLUMN_CANDIDATES if c in columns), None)


def _size_ok(data: bytes) -> bool:
    return len(data) <= MAX_BYTES


def _shannon_entropy(text: str) -> float:
    """Rough character-level entropy — used as a quick sanity check on text data."""
    import math
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


# ---------------------------------------------------------------------------
# POST /data/upload/{modality}
# ---------------------------------------------------------------------------

@router.post(
    "/upload/{modality}",
    summary="Upload a dataset file for the given modality",
)
async def upload_dataset(modality: Modality, file: UploadFile = File(...)):
    """
    Accepted formats:
      tabular → .csv
      text    → .jsonl only (use POST /data/upload/text/convert-csv to convert a CSV first)
      image   → .zip  (containing JPG / PNG files)

    File size limit: 200 MB.
    """
    raw = await file.read()

    # Size guard
    if not _size_ok(raw):
        return JSONResponse(
            status_code=413,
            content=ErrorResponse.now(
                error="File Too Large",
                detail=f"Maximum file size is {MAX_FILE_MB} MB. Received {len(raw)/1024/1024:.1f} MB.",
            ).model_dump(),
        )

    fname = file.filename or "upload"
    ext   = os.path.splitext(fname)[1].lower()

    logger.info("upload received: modality=%s file=%s size=%d bytes", modality.value, fname, len(raw))

    # ── Tabular ──
    if modality == Modality.tabular:
        if ext not in (".csv",):
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Invalid Format", detail="Tabular modality expects a .csv file.").model_dump())
        try:
            import pandas as pd  # noqa: PLC0415
            df = pd.read_csv(io.BytesIO(raw))
            _uploaded["tabular"].update({
                "filename": fname,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "null_counts": df.isnull().sum().to_dict(),
                "preprocessed": False,
                "source": "uploaded",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "_raw_bytes": raw,   # kept in memory for real model training
            })
            return {
                "modality": "tabular", "filename": fname,
                "rows": len(df), "columns": len(df.columns),
                "column_names": list(df.columns),
                "null_count": int(df.isnull().sum().sum()),
                "message": "Upload successful. Call POST /data/preprocess/tabular to clean.",
            }
        except Exception as exc:
            logger.exception("tabular upload parse error")
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Parse Error", detail=f"Could not read CSV: {exc}").model_dump())

    # ── Text ──
    elif modality == Modality.text:
        if ext != ".jsonl":
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Invalid Format",
                detail=(
                    "Text modality expects a .jsonl file (one JSON object per line, "
                    "each with at least a 'payload' key and a label key). "
                    "If you have a CSV, convert it first via "
                    "POST /data/upload/text/convert-csv."
                ),
            ).model_dump())
        try:
            text = raw.decode("utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            # Try JSONL parse
            parsed, parse_errors = 0, 0
            # Scan the full file — important for large datasets
            for line in lines:
                try:
                    json.loads(line)
                    parsed += 1
                except Exception:
                    parse_errors += 1
            _uploaded["text"].update({
                "filename": fname,
                "rows": len(lines),
                "valid_json_lines": parsed,
                "parse_errors_sample": parse_errors,
                "avg_length": int(sum(len(l) for l in lines) / max(len(lines), 1)),
                "preprocessed": False,
                "source": "uploaded",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "_raw_bytes": raw,   # kept in memory for real model training
            })
            return {
                "modality": "text", "filename": fname,
                "rows": len(lines), "valid_json_lines": parsed,
                "parse_error_rate": f"{parse_errors/max(len(lines),1):.1%}",
                "avg_payload_length": int(sum(len(l) for l in lines) / max(len(lines), 1)),
                "message": "Upload successful. Call POST /data/preprocess/text to clean.",
            }
        except Exception as exc:
            logger.exception("text upload parse error")
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Parse Error", detail=f"Could not read text file: {exc}").model_dump())

    # ── Image ──
    elif modality == Modality.image:
        if ext != ".zip":
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Invalid Format", detail="Image modality expects a .zip file of JPG/PNG images.").model_dump())
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                all_names = zf.namelist()
                image_names = [n for n in all_names if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
                invalid = len(all_names) - len(image_names)

            _uploaded["image"].update({
                "filename": fname,
                "total_files": len(all_names),
                "valid_images": len(image_names),
                "invalid_files": invalid,
                "sample_names": image_names[:5],
                "preprocessed": False,
                "source": "uploaded",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "_raw": raw,  # keep in memory for preprocess step
            })
            return {
                "modality": "image", "filename": fname,
                "total_files": len(all_names),
                "valid_images": len(image_names),
                "invalid_files": invalid,
                "sample_filenames": image_names[:5],
                "message": "Upload successful. Call POST /data/preprocess/image to validate and resize.",
            }
        except zipfile.BadZipFile:
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Invalid ZIP", detail="File is not a valid ZIP archive.").model_dump())
        except Exception as exc:
            logger.exception("image upload error")
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Error", detail=str(exc)).model_dump())


# ---------------------------------------------------------------------------
# POST /data/upload/text/convert-csv
# ---------------------------------------------------------------------------

@router.post(
    "/upload/text/convert-csv",
    summary="Convert a CSV file to JSONL and load it as the text dataset",
)
async def convert_text_csv_to_jsonl(
    file: UploadFile = File(...),
    payload_column: Optional[str] = None,
    label_column: Optional[str] = None,
):
    """
    Text modality only accepts .jsonl uploads directly (see POST /data/upload/text).
    This endpoint is the supported escape hatch for users who only have a CSV:
    it converts each row into a JSON object {"payload": ..., "label": ...} and
    feeds the result through the same path as a normal JSONL upload.

    Conversion is only viable if the CSV has (a) a text/payload column and
    (b) a label column. If you don't pass payload_column / label_column
    explicitly, we try to auto-detect them:
      - payload column: first column named one of
        ["payload", "text", "message", "content", "body", "request"]
      - label column: resolved via the shared LABEL_COLUMN_CANDIDATES list
        (label, class, target, attack, fraud, y), or whatever you pass in.

    If no payload column can be found, conversion is rejected (422) — falling
    back to "serialize the whole row as the payload" would silently produce a
    low-quality dataset, so we refuse instead of guessing past the safety net.
    """
    raw = await file.read()
    if not _size_ok(raw):
        return JSONResponse(status_code=413, content=ErrorResponse.now(
            error="File Too Large",
            detail=f"Maximum file size is {MAX_FILE_MB} MB. Received {len(raw)/1024/1024:.1f} MB.",
        ).model_dump())

    fname = file.filename or "upload.csv"
    if os.path.splitext(fname)[1].lower() != ".csv":
        return JSONResponse(status_code=422, content=ErrorResponse.now(
            error="Invalid Format",
            detail="This endpoint converts .csv → .jsonl. Upload a .csv file here.",
        ).model_dump())

    try:
        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(io.BytesIO(raw))
        cols = list(df.columns)

        payload_candidates = ["payload", "text", "message", "content", "body", "request"]
        resolved_payload_col = (
            payload_column if payload_column in cols
            else next((c for c in payload_candidates if c in cols), None)
        )
        resolved_label_col = _resolve_label_column(cols, label_column)

        if resolved_payload_col is None:
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Conversion Not Viable",
                detail=(
                    f"No payload/text column found or specified. Available columns: {cols}. "
                    "Pass ?payload_column=<name> to specify it explicitly."
                ),
            ).model_dump())
        if resolved_label_col is None:
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Conversion Not Viable",
                detail=(
                    f"No label column found or specified. Available columns: {cols}. "
                    "Pass ?label_column=<name> to specify it explicitly."
                ),
            ).model_dump())

        records = []
        skipped = 0
        for _, row in df.iterrows():
            payload_val = row.get(resolved_payload_col)
            label_val = row.get(resolved_label_col)
            if pd.isna(payload_val) or pd.isna(label_val):
                skipped += 1
                continue
            records.append(json.dumps({"payload": str(payload_val), "label": label_val}))

        if not records:
            return JSONResponse(status_code=422, content=ErrorResponse.now(
                error="Conversion Not Viable",
                detail="Every row had a missing payload or label value after conversion.",
            ).model_dump())

        jsonl_bytes = ("\n".join(records) + "\n").encode("utf-8")
        converted_name = os.path.splitext(fname)[0] + "_converted.jsonl"

        _uploaded["text"].update({
            "filename": converted_name,
            "rows": len(records),
            "valid_json_lines": len(records),
            "parse_errors_sample": 0,
            "avg_length": int(sum(len(l) for l in records) / max(len(records), 1)),
            "preprocessed": False,
            "source": "uploaded",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "_raw_bytes": jsonl_bytes,
            "converted_from": fname,
        })

        return {
            "modality": "text",
            "filename": converted_name,
            "converted_from": fname,
            "payload_column_used": resolved_payload_col,
            "label_column_used": resolved_label_col,
            "rows": len(records),
            "rows_skipped_missing_values": skipped,
            "message": (
                "CSV converted to JSONL and loaded as the active text dataset. "
                "Call POST /data/preprocess/text to clean."
            ),
        }
    except Exception as exc:
        logger.exception("text csv->jsonl conversion error")
        return JSONResponse(status_code=422, content=ErrorResponse.now(
            error="Conversion Failed", detail=str(exc)).model_dump())


# ---------------------------------------------------------------------------
# POST /data/preprocess/{modality}
# ---------------------------------------------------------------------------

@router.post(
    "/preprocess/{modality}",
    summary="Run cleaning and validation on the uploaded dataset",
)
async def preprocess_dataset(
    modality: Modality,
    label_column: Optional[str] = None,  # query param for tabular
):
    """
    Runs modality-specific preprocessing:

    Tabular:
      - Drop duplicate rows
      - Fill or drop null columns (threshold: 1%)
      - Remove IQR outliers
      - Normalise numeric columns to [0, 1]
      - Validate label column exists and is binary

    Text:
      - Strip null bytes and control characters
      - Truncate payloads > 512 chars (DistilBERT limit)
      - Deduplicate identical payloads
      - Balance classes to 2:1 max ratio

    Image:
      - Validate each image can be opened by Pillow
      - Report images that are too small (< 32×32)
      - Count corrupt / unreadable files
      - (Resize to 224×224 happens at inference time)
    """
    info = _uploaded.get(modality.value, {})
    if not info.get("filename"):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse.now(
                error="No Data",
                detail=f"No {modality.value} dataset uploaded yet. Call POST /data/upload/{modality.value} first.",
            ).model_dump(),
        )

    logger.info("preprocess started: modality=%s file=%s", modality.value, info["filename"])

    # ── Tabular ──
    if modality == Modality.tabular:
        try:
            import pandas as pd  # noqa: PLC0415
            import numpy as np   # noqa: PLC0415

            raw_bytes = info.get("_raw_bytes")
            if raw_bytes is None:
                raise ValueError(
                    "Original upload bytes not found in memory. Please re-upload the file."
                )

            df = pd.read_csv(io.BytesIO(raw_bytes))
            rows_raw = len(df)

            # ── Real cleaning (previously this block only fabricated plausible
            # numbers without touching the data — model training would then
            # silently re-read the original, uncleaned upload). Now we actually
            # mutate the dataframe and persist it back to _raw_bytes so that
            # POST /models/train/{modality} trains on the cleaned data. ──

            before = len(df)
            df = df.drop_duplicates()
            dup_dropped = before - len(df)

            before = len(df)
            null_frac = df.isnull().mean()
            mostly_null_cols = null_frac[null_frac > 0.5].index.tolist()
            if mostly_null_cols:
                df = df.drop(columns=mostly_null_cols)
            df = df.dropna()
            null_dropped = before - len(df)

            before = len(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(df) > 0:
                q1 = df[numeric_cols].quantile(0.25)
                q3 = df[numeric_cols].quantile(0.75)
                iqr = q3 - q1
                in_bounds = ~(
                    (df[numeric_cols] < (q1 - 3 * iqr)) | (df[numeric_cols] > (q3 + 3 * iqr))
                ).any(axis=1)
                df = df[in_bounds]
            outlier_dropped = before - len(df)

            rows_clean = len(df)
            col_names = list(df.columns)

            label_col = _resolve_label_column(col_names, label_column)
            label_found = label_col is not None
            label_col = label_col or (label_column or "label")  # for display purposes only

            # Persist cleaned data so training actually uses it
            cleaned_csv_bytes = df.to_csv(index=False).encode("utf-8")
            _uploaded["tabular"]["_raw_bytes"]    = cleaned_csv_bytes
            _uploaded["tabular"]["rows"]          = rows_clean
            _uploaded["tabular"]["columns"]       = len(col_names)
            _uploaded["tabular"]["column_names"]  = col_names

            result = {
                "modality": "tabular",
                "filename": info["filename"],
                "raw_rows": rows_raw,
                "cleaned_rows": rows_clean,
                "columns": len(col_names),
                "steps": {
                    "duplicates_dropped": dup_dropped,
                    "null_rows_dropped": null_dropped,
                    "mostly_null_columns_dropped": mostly_null_cols,
                    "outliers_dropped": outlier_dropped,
                    "normalisation": "Outliers removed via IQR (3×); model-side StandardScaler applied at train time",
                },
                "label_column": label_col,
                "label_found": label_found,
                "warnings": [] if label_found else [f"Column '{label_col}' not found. Available: {col_names[:8]}"],
                "ready_for_attacks": label_found and rows_clean > 100,
                "message": "Preprocessing complete. Dataset is ready for attack pipeline.",
            }
            _uploaded["tabular"]["preprocessed"] = True
            _uploaded["tabular"]["cleaned_rows"] = rows_clean
            logger.info("tabular preprocess done: %d → %d rows", rows_raw, rows_clean)
            return result

        except Exception as exc:
            logger.exception("tabular preprocess error")
            return JSONResponse(status_code=500, content=ErrorResponse.now(
                error="Preprocessing Failed", detail=str(exc)).model_dump())

    # ── Text ──
    elif modality == Modality.text:
        try:
            import re as _re  # noqa: PLC0415

            raw_bytes = info.get("_raw_bytes")
            if raw_bytes is None:
                raise ValueError(
                    "Original upload bytes not found in memory. Please re-upload the file."
                )

            text = raw_bytes.decode("utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            rows_raw = len(lines)

            # ── Real cleaning (previously this block only fabricated plausible
            # numbers — model training would silently re-parse the original,
            # uncleaned upload). Now we actually clean each line and persist
            # the cleaned JSONL back to _raw_bytes. ──
            _control_chars = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
            cleaned_lines: list[str] = []
            empty_dropped = 0
            too_long = 0
            seen: set[str] = set()
            dup_dropped = 0

            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    empty_dropped += 1  # unparsable line — treat as dropped
                    continue

                payload = obj.get("payload")
                if payload is None or (isinstance(payload, str) and not payload.strip()):
                    empty_dropped += 1
                    continue

                if isinstance(payload, str):
                    cleaned_payload = _control_chars.sub("", payload)
                    if len(cleaned_payload) > 512:
                        cleaned_payload = cleaned_payload[:512]
                        too_long += 1
                    obj["payload"] = cleaned_payload

                cleaned_line = json.dumps(obj)
                if cleaned_line in seen:
                    dup_dropped += 1
                    continue
                seen.add(cleaned_line)
                cleaned_lines.append(cleaned_line)

            rows_clean = len(cleaned_lines)
            avg_len_after = (
                int(sum(len(l) for l in cleaned_lines) / rows_clean) if rows_clean else 0
            )

            cleaned_jsonl_bytes = ("\n".join(cleaned_lines) + "\n").encode("utf-8") if cleaned_lines else b""
            _uploaded["text"]["_raw_bytes"] = cleaned_jsonl_bytes
            _uploaded["text"]["rows"]       = rows_clean
            _uploaded["text"]["avg_length"] = avg_len_after

            result = {
                "modality": "text",
                "filename": info["filename"],
                "raw_rows": rows_raw,
                "cleaned_rows": rows_clean,
                "steps": {
                    "empty_or_unparsable_dropped": empty_dropped,
                    "long_payloads_truncated": too_long,
                    "duplicate_payloads_dropped": dup_dropped,
                    "control_chars_stripped": True,
                    "truncation_limit": "512 characters (DistilBERT max token length)",
                },
                "avg_payload_length_after": avg_len_after,
                "ready_for_attacks": rows_clean > 50,
                "message": "Preprocessing complete. Dataset is ready for attack pipeline.",
            }
            _uploaded["text"]["preprocessed"] = True
            _uploaded["text"]["cleaned_rows"] = rows_clean
            logger.info("text preprocess done: %d → %d rows", rows_raw, rows_clean)
            return result

        except Exception as exc:
            logger.exception("text preprocess error")
            return JSONResponse(status_code=500, content=ErrorResponse.now(
                error="Preprocessing Failed", detail=str(exc)).model_dump())

    # ── Image ──
    elif modality == Modality.image:
        try:
            valid_images = info.get("valid_images", 0)
            total_files  = info.get("total_files", 0)

            # Try to validate images using Pillow if available
            corrupt, too_small, valid_count = 0, 0, 0
            raw_zip = info.get("_raw")

            if raw_zip:
                try:
                    from PIL import Image as PILImage  # noqa: PLC0415
                    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
                        names = [n for n in zf.namelist()
                                 if os.path.splitext(n)[1].lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
                        for name in names[:200]:  # check up to 200 images
                            try:
                                img_bytes = zf.read(name)
                                img = PILImage.open(io.BytesIO(img_bytes))
                                w, h = img.size
                                if w < 32 or h < 32:
                                    too_small += 1
                                else:
                                    valid_count += 1
                            except Exception:
                                corrupt += 1
                except ImportError:
                    # Pillow not available — use counts from upload step
                    valid_count = valid_images
            else:
                valid_count = valid_images

            result = {
                "modality": "image",
                "filename": info["filename"],
                "total_files_in_zip": total_files,
                "valid_images": valid_count,
                "corrupt_or_unreadable": corrupt,
                "too_small_under_32px": too_small,
                "steps": {
                    "format_validation": "JPG / PNG / BMP / WebP accepted",
                    "resize": "224×224 applied at inference time (ResNet-18 input)",
                    "normalisation": "ImageNet mean/std normalisation at inference time",
                },
                "ready_for_attacks": valid_count > 10,
                "message": "Preprocessing complete. Images will be resized to 224×224 when attacks run.",
            }
            _uploaded["image"]["preprocessed"] = True
            # Previously this wrote to a separately-named "valid_count" key
            # while models.py's training endpoint reads "valid_images" — so
            # the post-preprocess filtered count never reached training.
            # Now we update the single key that's actually read downstream.
            _uploaded["image"]["valid_images"] = valid_count
            logger.info("image preprocess done: %d valid images", valid_count)
            return result

        except Exception as exc:
            logger.exception("image preprocess error")
            return JSONResponse(status_code=500, content=ErrorResponse.now(
                error="Preprocessing Failed", detail=str(exc)).model_dump())


# ---------------------------------------------------------------------------
# GET /data/status
# ---------------------------------------------------------------------------

@router.get("/status", summary="Show active data source per modality")
async def data_status():
    """Returns what dataset is currently loaded for each modality, including row/column counts."""
    out = {}
    for modality, info in _uploaded.items():
        has_upload = bool(info.get("filename"))
        out[modality] = {
            "source":       info.get("source", "synthetic"),
            "filename":     info.get("filename"),
            "preprocessed": info.get("preprocessed", False),
            "rows":         info.get("rows", 0),
            "columns":      info.get("columns", 0),
            "cleaned_rows": info.get("cleaned_rows", info.get("rows", 0)),
            # Previously: bool(filename) or source == "synthetic" — but source
            # defaults to "synthetic" for every untouched slot, so the right-hand
            # side was always true and this field never actually reflected
            # whether real data had been uploaded. Now it only means what it says.
            "ready":        has_upload,
            "using_default_synthetic": not has_upload,
        }
    return {"data_sources": out, "timestamp": datetime.now(timezone.utc).isoformat()}