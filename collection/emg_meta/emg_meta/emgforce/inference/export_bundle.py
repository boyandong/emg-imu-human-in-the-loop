from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_LABELS = {
    "model_outputs": [
        "index_press",
        "index_release",
        "middle_press",
        "middle_release",
        "thumb_click",
        "thumb_down",
        "thumb_in",
        "thumb_out",
        "thumb_up",
    ],
    "display_names": {
        "index_press": "食指按下",
        "index_release": "食指释放",
        "middle_press": "中指按下",
        "middle_release": "中指释放",
        "thumb_click": "拇指轻点",
        "thumb_down": "拇指向下",
        "thumb_in": "拇指向内",
        "thumb_out": "拇指向外",
        "thumb_up": "拇指向上",
    },
}

DEFAULT_PREPROCESSING = {
    "preprocessing_version": "meta_8ch_v1",
    "realtime_buffer_seconds": 4.0,
    "realtime_model_window_seconds": 2.0,
    "realtime_fixed_lag_seconds": 0.25,
    "online_event_threshold": 0.50,
    "offline_cler_threshold": 0.35,
    "debounce_seconds": 0.05,
    "minimum_hold_seconds": 0.50,
}


def create_bundle_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_id: str | None = None,
    channels: int = 8,
    sample_rate: int = 2000,
) -> Path:
    """Package a trained .ckpt or .ts artifact into an EMGForce model bundle."""
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到权重文件：{ckpt}")

    if model_id is None:
        model_id = ckpt.stem

    if output_dir is None:
        models_root = Path(__file__).resolve().parent.parent.parent / "models"
        bundle_dir = models_root / model_id
    else:
        bundle_dir = Path(output_dir).resolve() / model_id

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制或处理 artifact
    target_artifact = bundle_dir / ckpt.name
    if ckpt != target_artifact:
        shutil.copy2(ckpt, target_artifact)

    sha256 = hashlib.sha256(target_artifact.read_bytes()).hexdigest()

    # 2. 处理 config
    target_config = bundle_dir / "training_config.yaml"
    if config_path and Path(config_path).is_file():
        shutil.copy2(Path(config_path), target_config)
    elif not target_config.is_file():
        target_config.write_text(
            f"network:\n  input_channels: {channels}\n  output_channels: 9\n",
            encoding="utf-8",
        )

    # 3. 标签与预处理文件
    labels_file = bundle_dir / "labels.json"
    if not labels_file.is_file():
        labels_file.write_text(
            json.dumps(DEFAULT_LABELS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    prep_file = bundle_dir / "preprocessing.json"
    if not prep_file.is_file():
        prep_file.write_text(
            json.dumps(DEFAULT_PREPROCESSING, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    is_ckpt = target_artifact.suffix == ".ckpt"
    manifest: dict[str, Any] = {
        "bundle_version": 2,
        "model_id": model_id,
        "algorithm_id": "meta_conv_lstm_v1",
        "runtime_backend": "lightning_conv_lstm",
        "task": "discrete_gestures",
        "artifact": {
            "filename": target_artifact.name,
            "format": "pytorch_lightning" if is_ckpt else "torchscript",
            "sha256": sha256,
            "size_bytes": target_artifact.stat().st_size,
        },
        "network": {
            "input_channels": channels,
            "output_channels": len(DEFAULT_LABELS["model_outputs"]),
            "sample_rate_hz": sample_rate,
            "conv_kernel_width": 21,
            "stride": 10,
            "left_context_samples": 20,
        },
        "signal": {"input_channels": channels, "sample_rate_hz": sample_rate},
        "preprocessing": DEFAULT_PREPROCESSING,
        "features": {},
        "decoder": {},
        "training": {
            "fitness_note": "User trained custom model bundle",
        },
        "labels_file": "labels.json",
        "preprocessing_file": "preprocessing.json",
        "training_config_file": "training_config.yaml",
    }

    manifest_file = bundle_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[+] 成功构建模型包：{bundle_dir}")
    print(f"    - 模型ID: {model_id}")
    print(f"    - 通道数: {channels}, 采样率: {sample_rate}Hz")
    print(f"    - 权重文件: {target_artifact.name}")
    return bundle_dir


def main():
    parser = argparse.ArgumentParser(description="将训练好的模型打包为上位机 Model Bundle")
    parser.add_argument("checkpoint", help="模型权重文件路径 (.ckpt 或 .ts)")
    parser.add_argument("--config", help="配置文件路径 (training_config.yaml 或 model_config.yaml)", default=None)
    parser.add_argument("--output-dir", help="输出目录 (默认为 models/ 文件夹)", default=None)
    parser.add_argument("--model-id", help="自定义模型 ID 名称", default=None)
    parser.add_argument("--channels", type=int, default=8, help="手环输入通道数 (默认 8)")
    parser.add_argument("--sample-rate", type=int, default=2000, help="采样率 (默认 2000Hz)")

    args = parser.parse_args()
    create_bundle_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        model_id=args.model_id,
        channels=args.channels,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
