"""Fine-tune wav2vec2 on TIMIT, following the HF blog, for two tasks.

    # step 1 -- the blog's experiment: character-level ASR, scored by WER
    python src/finetune.py --task asr --epochs 30

    # step 2 -- phoneme recognition, scored by PER
    python src/finetune.py --task phoneme --epochs 30

Both tasks share this file because the recipe is identical: the only differences are
what the labels are (characters vs phones) and how the output is scored.

Adapted from the blog for transformers 5.x, which removed or renamed several of the APIs
it uses -- see ADAPTATIONS below.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

# MPS has no aten::_ctc_loss kernel, and Wav2Vec2ForCTC computes the loss inside its
# own forward, so the loss cannot simply be moved to CPU as it can with a hand-written
# training loop. This lets that single op fall back to CPU while everything else stays
# on the GPU. Must be set before torch initialises.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from timit import PhoneCoder, find_root, load_split, phone_inventory

# ADAPTATIONS from the blog (written for transformers 4.x):
#   processor.as_target_processor()  -> removed; call the tokenizer directly
#   model.freeze_feature_extractor() -> renamed freeze_feature_encoder()
#   TrainingArguments.evaluation_strategy -> renamed eval_strategy
#   TrainingArguments.group_by_length -> removed; we sort by length ourselves
#   datasets.load_metric -> removed; jiwer computes WER/PER directly
#   fp16 -> unsupported on MPS, so training runs in fp32 there


class TimitDataset(Dataset):
    """Yields raw waveform + label ids. Audio is decoded lazily in the workers."""

    def __init__(self, utterances, processor, task: str, coder: PhoneCoder | None):
        self.utterances = utterances
        self.processor = processor
        self.task = task
        self.coder = coder

    def __len__(self) -> int:
        return len(self.utterances)

    def target_text(self, utt) -> str:
        return utt.text if self.task == "asr" else self.coder.encode(utt.phones)

    def __getitem__(self, index):
        utt = self.utterances[index]
        speech, sample_rate = sf.read(utt.path, dtype="float32")
        values = self.processor(speech, sampling_rate=sample_rate).input_values[0]
        labels = self.processor.tokenizer(self.target_text(utt)).input_ids
        return {"input_values": values, "labels": labels}


@dataclass
class DataCollatorCTCWithPadding:
    """The blog's collator: pad inputs and labels separately, mask label padding.

    Label padding becomes -100 so the CTC loss ignores it.
    """

    processor: object
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding,
                                   return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding,
                                                    return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_vocab(train, test, task: str, coder: PhoneCoder | None, out_path: str) -> dict:
    """Blog step: derive the vocabulary from the transcriptions themselves."""
    if task == "asr":
        text = " ".join(u.text for u in train) + " ".join(u.text for u in test)
        vocab_list = sorted(set(text))
        vocab = {v: k for k, v in enumerate(vocab_list)}
        # The blog's trick: make the space visible as '|' so word boundaries survive.
        vocab["|"] = vocab[" "]
        del vocab[" "]
    else:
        # One symbol per phone; no word delimiter is meaningful for phone sequences.
        vocab = {coder.to_char[p]: i for i, p in enumerate(coder.phones)}

    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(vocab, fh)
    return vocab


def make_metrics(processor, task: str, coder: PhoneCoder | None):
    import jiwer

    def compute_metrics(pred):
        logits = pred.predictions
        pred_ids = np.argmax(logits, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        if task == "asr":
            keep = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not keep:
                return {"wer": 1.0}
            preds, refs = zip(*keep)
            return {"wer": jiwer.wer(list(refs), list(preds))}

        # Phone error rate: decode private-use chars back to phone names and score the
        # sequences as space-separated "words", which makes jiwer's edit distance the
        # standard PER.
        errors = length = 0
        for p, l in zip(pred_str, label_str):
            ref = coder.decode(l)
            hyp = coder.decode(p)
            if not ref:
                continue
            out = jiwer.process_words([" ".join(ref)], [" ".join(hyp)])
            errors += out.substitutions + out.deletions + out.insertions
            length += len(ref)
        return {"per": errors / max(length, 1)}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["asr", "phoneme"], required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--epochs", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4, help="8 x 4 = the blog's 32")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-split", default="test", choices=["test", "core_test"])
    parser.add_argument("--max-train", type=int, default=None, help="subset, for smoke runs")
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                              Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                              Wav2Vec2Processor)

    root = find_root(".")
    train = load_split(root, "train")
    evaluation = load_split(root, args.eval_split)
    if args.max_train:
        train = train[:args.max_train]
    if args.max_eval:
        evaluation = evaluation[:args.max_eval]
    print(f"TIMIT: {len(train)} train / {len(evaluation)} {args.eval_split}")

    coder = None
    if args.task == "phoneme":
        coder = PhoneCoder(phone_inventory([train, evaluation]))
        print(f"phone inventory: {len(coder.phones)} folded phones")

    output_dir = args.output_dir or f"exp/wav2vec2-timit-{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    vocab = build_vocab(train, evaluation, args.task, coder, vocab_path)
    print(f"vocab: {len(vocab)} tokens -> {vocab_path}")

    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
        return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    # Sorting by duration replaces the removed group_by_length: batches of similar
    # length waste far less compute on padding.
    train = sorted(train, key=lambda u: os.path.getsize(u.path))

    train_ds = TimitDataset(train, processor, args.task, coder)
    eval_ds = TimitDataset(evaluation, processor, args.task, coder)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    # The convolutional feature encoder is already good; the blog freezes it.
    model.freeze_feature_encoder()

    use_mps = torch.backends.mps.is_available() and args.device in ("auto", "mps")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=False,                      # unsupported on MPS
        learning_rate=args.lr,
        weight_decay=0.005,
        warmup_steps=args.warmup,
        logging_steps=25,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=2,
        use_cpu=not use_mps and not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=make_metrics(processor, args.task, coder),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor.feature_extractor,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nfinal:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
