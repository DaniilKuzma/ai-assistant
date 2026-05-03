"""Train the final hybrid edit-and-punctuation corrector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from candidate_generator import CandidateGenerator
from edit_labels import build_examples_from_dataframe
from hybrid_preprocessor import HybridPreprocessor
from training_augmentation import (
    augment_keep_candidates,
    build_clean_candidate_audit,
    filter_noisy_clean_rows,
    populate_top_k_candidates,
)


def load_splits(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dataset_path)
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        clean = df["correct_text"].drop_duplicates().sample(frac=1.0, random_state=42).tolist()
        n = len(clean)
        val_set = set(clean[: int(n * 0.1)])
        test_set = set(clean[int(n * 0.1) : int(n * 0.2)])
        train_df = df[~df["correct_text"].isin(val_set | test_set)].copy()
        val_df = df[df["correct_text"].isin(val_set)].copy()
        test_df = df[df["correct_text"].isin(test_set)].copy()
    return train_df, val_df, test_df


def train_hybrid(
    dataset_path: str = "data/processed/dataset.csv",
    output_dir: str = "models",
    max_length: int = 128,
    max_vocab_size: int = 80_000,
    batch_size: int = 64,
    epochs: int = 30,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-4,
    candidate_min_freq: int = 1,
    candidate_max_distance: int = 1,
    candidate_top_k: int = 8,
    safe_split_candidates: bool = True,
    long_oov_max_distance: int = 2,
    long_oov_min_length: int = 8,
    min_dictionary_score: float = 0.25,
    keep_candidate_augmentation: bool = True,
    keep_candidate_probability: float = 0.35,
    keep_candidate_max_per_example: int = 3,
    keep_candidate_max_searches: int = 8_000,
    keep_candidate_clean_only: bool = True,
    clean_audit_max_word_searches: int = 5_000,
    clean_action_keep_weight: float = 2.0,
    dirty_action_keep_weight: float = 1.0,
    action_change_weight: float = 3.0,
    punct_change_weight: float = 8.0,
    final_punct_weight: float = 8.0,
    clean_punct_keep_weight: float = 2.0,
    dirty_punct_keep_weight: float = 1.0,
    context_reranker_enabled: bool = True,
    context_model_name: str = "DeepPavlov/rubert-base-cased",
    context_device: str = "cpu",
    context_margin: float = 0.25,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report", exist_ok=True)

    train_df, val_df, test_df = load_splits(dataset_path)
    provisional_generator = CandidateGenerator.from_texts(
        train_df["correct_text"].dropna().astype(str).tolist(),
        min_freq=candidate_min_freq,
        max_distance=candidate_max_distance,
        allow_split_candidates=False,
        safe_split_only=True,
        long_oov_max_distance=long_oov_max_distance,
        long_oov_min_length=long_oov_min_length,
    )
    all_rows = pd.concat([train_df, val_df, test_df], ignore_index=True)
    filtered_rows, noise_audit, suspicious_texts = filter_noisy_clean_rows(
        all_rows,
        provisional_generator,
        min_dictionary_score=min_dictionary_score,
        max_word_searches=clean_audit_max_word_searches,
    )
    noise_audit_path = Path("report") / "clean_noise_filter.csv"
    noise_audit.to_csv(noise_audit_path, index=False, encoding="utf-8")
    if suspicious_texts:
        train_df = train_df.loc[~train_df["correct_text"].astype(str).isin(suspicious_texts)].copy()
        val_df = val_df.loc[~val_df["correct_text"].astype(str).isin(suspicious_texts)].copy()
        test_df = test_df.loc[~test_df["correct_text"].astype(str).isin(suspicious_texts)].copy()
    print(f"Clean noise filter rows: {len(noise_audit)} -> {noise_audit_path}")

    train_df.to_csv("data/processed/train.csv", index=False, encoding="utf-8")
    val_df.to_csv("data/processed/val.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/processed/test.csv", index=False, encoding="utf-8")

    print(f"Train/val/test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")

    candidate_generator = CandidateGenerator.from_texts(
        train_df["correct_text"].dropna().astype(str).tolist(),
        min_freq=candidate_min_freq,
        max_distance=candidate_max_distance,
        allow_split_candidates=safe_split_candidates,
        safe_split_only=True,
        long_oov_max_distance=long_oov_max_distance,
        long_oov_min_length=long_oov_min_length,
    )
    candidate_path = str(Path(output_dir) / "candidate_generator.pkl")
    candidate_generator.save(candidate_path)

    audit = build_clean_candidate_audit(
        pd.concat([train_df, val_df, test_df], ignore_index=True),
        candidate_generator,
        min_dictionary_score=min_dictionary_score,
        max_word_searches=clean_audit_max_word_searches,
    )
    audit_path = Path("report") / "clean_candidate_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    print(f"Clean candidate audit rows: {len(audit)} -> {audit_path}")

    train_examples, train_stats = build_examples_from_dataframe(train_df, max_tokens=max_length)
    val_examples, val_stats = build_examples_from_dataframe(val_df, max_tokens=max_length)
    print(f"Hybrid train examples: {train_stats}")
    print(f"Hybrid val examples:   {val_stats}")

    if not train_examples or not val_examples:
        raise RuntimeError("Not enough aligned examples for hybrid training.")

    train_examples, train_topk_stats = populate_top_k_candidates(
        train_examples,
        candidate_generator,
        candidate_top_k=candidate_top_k,
    )
    val_examples, val_topk_stats = populate_top_k_candidates(
        val_examples,
        candidate_generator,
        candidate_top_k=candidate_top_k,
    )
    print(f"Top-k train candidate stats: {train_topk_stats}")
    print(f"Top-k val candidate stats:   {val_topk_stats}")

    train_examples, train_keep_stats = augment_keep_candidates(
        train_examples,
        candidate_generator,
        enabled=keep_candidate_augmentation,
        probability=keep_candidate_probability,
        max_per_example=keep_candidate_max_per_example,
        max_candidate_searches=keep_candidate_max_searches,
        clean_only=keep_candidate_clean_only,
        seed=42,
        min_dictionary_score=min_dictionary_score,
    )
    val_examples, val_keep_stats = augment_keep_candidates(
        val_examples,
        candidate_generator,
        enabled=keep_candidate_augmentation,
        probability=keep_candidate_probability,
        max_per_example=keep_candidate_max_per_example,
        max_candidate_searches=max(1, keep_candidate_max_searches // 4),
        clean_only=keep_candidate_clean_only,
        seed=43,
        min_dictionary_score=min_dictionary_score,
    )
    print(f"Candidate-aware train KEEP stats: {train_keep_stats}")
    print(f"Candidate-aware val KEEP stats:   {val_keep_stats}")

    preprocessor = HybridPreprocessor(
        max_length=max_length,
        min_freq=1,
        max_vocab_size=max_vocab_size,
        candidate_top_k=candidate_top_k,
    )
    preprocessor.fit(train_examples)
    preprocessor_path = str(Path(output_dir) / "hybrid_preprocessor.pkl")
    preprocessor.save(preprocessor_path)

    x_train, y_train, sw_train = preprocessor.vectorize_examples(
        train_examples,
        clean_action_keep_weight=clean_action_keep_weight,
        dirty_action_keep_weight=dirty_action_keep_weight,
        action_change_weight=action_change_weight,
        clean_punct_keep_weight=clean_punct_keep_weight,
        dirty_punct_keep_weight=dirty_punct_keep_weight,
        punct_change_weight=punct_change_weight,
        final_punct_weight=final_punct_weight,
    )
    x_val, y_val, sw_val = preprocessor.vectorize_examples(
        val_examples,
        clean_action_keep_weight=clean_action_keep_weight,
        dirty_action_keep_weight=dirty_action_keep_weight,
        action_change_weight=action_change_weight,
        clean_punct_keep_weight=clean_punct_keep_weight,
        dirty_punct_keep_weight=dirty_punct_keep_weight,
        punct_change_weight=punct_change_weight,
        final_punct_weight=final_punct_weight,
    )

    from hybrid_model import build_hybrid_model
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model = build_hybrid_model(
        vocab_size=preprocessor.vocab_size,
        max_length=max_length,
        action_classes=preprocessor.action_classes,
        punct_classes=preprocessor.punct_classes,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        candidate_top_k=candidate_top_k,
    )
    model.summary()

    model_path = str(Path(output_dir) / "hybrid_corrector.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(str(Path(output_dir) / "hybrid_training_log.csv")),
    ]

    history = model.fit(
        x_train,
        y_train,
        sample_weight=sw_train,
        validation_data=(x_val, y_val, sw_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_path)
    config = {
        "model_version": 8,
        "requires_source_punct_ids": True,
        "requires_top_k_candidates": True,
        "requires_context_reranker": context_reranker_enabled,
        "dataset_path": dataset_path,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
        "candidate_generator_path": candidate_path,
        "max_length": max_length,
        "max_vocab_size": max_vocab_size,
        "vocab_size": preprocessor.vocab_size,
        "candidate_min_freq": candidate_min_freq,
        "candidate_max_distance": candidate_max_distance,
        "candidate_top_k": candidate_top_k,
        "safe_split_candidates": safe_split_candidates,
        "long_oov_max_distance": long_oov_max_distance,
        "long_oov_min_length": long_oov_min_length,
        "min_dictionary_score": min_dictionary_score,
        "keep_candidate_augmentation": keep_candidate_augmentation,
        "keep_candidate_probability": keep_candidate_probability,
        "keep_candidate_max_per_example": keep_candidate_max_per_example,
        "keep_candidate_max_searches": keep_candidate_max_searches,
        "keep_candidate_clean_only": keep_candidate_clean_only,
        "clean_audit_max_word_searches": clean_audit_max_word_searches,
        "clean_action_keep_weight": clean_action_keep_weight,
        "dirty_action_keep_weight": dirty_action_keep_weight,
        "action_change_weight": action_change_weight,
        "punct_change_weight": punct_change_weight,
        "final_punct_weight": final_punct_weight,
        "clean_punct_keep_weight": clean_punct_keep_weight,
        "dirty_punct_keep_weight": dirty_punct_keep_weight,
        "context_reranker_enabled": context_reranker_enabled,
        "context_model_name": context_model_name,
        "context_device": context_device,
        "context_margin": context_margin,
        "action_classes": preprocessor.action_classes,
        "punct_classes": preprocessor.punct_classes,
        "d_model": d_model,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_topk_candidate_stats": train_topk_stats,
        "val_topk_candidate_stats": val_topk_stats,
        "train_keep_candidate_stats": train_keep_stats,
        "val_keep_candidate_stats": val_keep_stats,
        "clean_candidate_audit_path": str(audit_path),
        "clean_noise_filter_path": str(noise_audit_path),
        "clean_noise_filter_rows": int(len(noise_audit)),
        "clean_noise_filter_texts": int(len(suspicious_texts)),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    with open(Path(output_dir) / "hybrid_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid corrector.")
    parser.add_argument("--dataset", default="data/processed/dataset.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--candidate-min-freq", type=int, default=1)
    parser.add_argument("--candidate-max-distance", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--disable-safe-split-candidates", action="store_true")
    parser.add_argument("--long-oov-max-distance", type=int, default=2)
    parser.add_argument("--long-oov-min-length", type=int, default=8)
    parser.add_argument("--min-dictionary-score", type=float, default=0.25)
    parser.add_argument("--disable-keep-candidate-augmentation", action="store_true")
    parser.add_argument("--keep-candidate-probability", type=float, default=0.35)
    parser.add_argument("--keep-candidate-max-per-example", type=int, default=3)
    parser.add_argument("--keep-candidate-max-searches", type=int, default=8_000)
    parser.add_argument("--keep-candidate-include-dirty", action="store_true")
    parser.add_argument("--clean-audit-max-word-searches", type=int, default=5_000)
    parser.add_argument("--clean-action-keep-weight", type=float, default=2.0)
    parser.add_argument("--dirty-action-keep-weight", type=float, default=1.0)
    parser.add_argument("--action-change-weight", type=float, default=3.0)
    parser.add_argument("--punct-change-weight", type=float, default=8.0)
    parser.add_argument("--final-punct-weight", type=float, default=8.0)
    parser.add_argument("--clean-punct-keep-weight", type=float, default=2.0)
    parser.add_argument("--dirty-punct-keep-weight", type=float, default=1.0)
    parser.add_argument("--disable-context-reranker", action="store_true")
    parser.add_argument("--context-model-name", default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--context-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--context-margin", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_hybrid(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        candidate_min_freq=args.candidate_min_freq,
        candidate_max_distance=args.candidate_max_distance,
        candidate_top_k=args.candidate_top_k,
        safe_split_candidates=not args.disable_safe_split_candidates,
        long_oov_max_distance=args.long_oov_max_distance,
        long_oov_min_length=args.long_oov_min_length,
        min_dictionary_score=args.min_dictionary_score,
        keep_candidate_augmentation=not args.disable_keep_candidate_augmentation,
        keep_candidate_probability=args.keep_candidate_probability,
        keep_candidate_max_per_example=args.keep_candidate_max_per_example,
        keep_candidate_max_searches=args.keep_candidate_max_searches,
        keep_candidate_clean_only=not args.keep_candidate_include_dirty,
        clean_audit_max_word_searches=args.clean_audit_max_word_searches,
        clean_action_keep_weight=args.clean_action_keep_weight,
        dirty_action_keep_weight=args.dirty_action_keep_weight,
        action_change_weight=args.action_change_weight,
        punct_change_weight=args.punct_change_weight,
        final_punct_weight=args.final_punct_weight,
        clean_punct_keep_weight=args.clean_punct_keep_weight,
        dirty_punct_keep_weight=args.dirty_punct_keep_weight,
        context_reranker_enabled=not args.disable_context_reranker,
        context_model_name=args.context_model_name,
        context_device=args.context_device,
        context_margin=args.context_margin,
    )


if __name__ == "__main__":
    main()
