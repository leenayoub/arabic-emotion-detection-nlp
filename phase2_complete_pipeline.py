"""Complete ArPanEmo Phase 2 pipeline in one file.

Stages: dataset inspection, EDA, preprocessing, feature extraction, majority
baseline, TF-IDF + Logistic Regression, AraBERT, evaluation, and error files.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import save_npz
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder


SEED = 42
LABELS = [
    "anger", "anticipation", "confusion", "disgust", "fear", "happiness",
    "neutral", "optimism", "pessimism", "sadness", "surprise",
]
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION = re.compile(r"@[A-Za-z0-9_]+")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def available(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text and not re.fullmatch(r"\.+", text) and re.search(r"[\u0600-\u06FFA-Za-z]", text))


def clean_text(value):
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL.sub(" URL ", text)
    text = MENTION.sub(" USER ", text)
    text = re.sub(r"#(?=\w)", "", text)
    text = DIACRITICS.sub("", text).replace("ـ", "")
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = "".join(
        ch if (ch.isalnum() or ch.isspace() or ch in "!?_" or ord(ch) > 0x1F000) else " "
        for ch in text
    )
    return re.sub(r"\s+", " ", text).strip()


def inspect(df, name):
    usable = df["post"].map(available)
    return {
        "split": name, "rows": len(df), "columns": df.columns.tolist(),
        "missing_post": int(df["post"].isna().sum()),
        "available_post": int(usable.sum()), "unavailable_post": int((~usable).sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "class_distribution": df["label"].value_counts().sort_index().astype(int).to_dict(),
    }


def preprocess(df):
    work = df.loc[df["post"].map(available)].copy()
    removed_unavailable = len(df) - len(work)
    work["clean_text"] = work["post"].map(clean_text)
    removed_empty = int(work["clean_text"].eq("").sum())
    work = work[work["clean_text"].ne("")].copy()
    counts = work.groupby("clean_text")["label"].nunique()
    conflicts = set(counts[counts > 1].index)
    removed_conflicts = int(work["clean_text"].isin(conflicts).sum())
    work = work[~work["clean_text"].isin(conflicts)].copy()
    duplicates = work.duplicated("clean_text", keep="first")
    removed_duplicates = int(duplicates.sum())
    work = work[~duplicates].reset_index(drop=True)
    return work, {
        "removed_unavailable": removed_unavailable, "removed_empty": removed_empty,
        "removed_conflicting_rows": removed_conflicts,
        "removed_duplicate_clean_text": removed_duplicates,
    }


def split_clean_train(df):
    counts = df["label"].value_counts()
    rare_labels = counts[counts < 2].index
    rare = df[df["label"].isin(rare_labels)]
    eligible = df[~df["label"].isin(rare_labels)]
    train, validation = train_test_split(
        eligible, test_size=0.20, random_state=SEED, stratify=eligible["label"]
    )
    train = pd.concat([train, rare], ignore_index=True)
    return train.sample(frac=1, random_state=SEED).reset_index(drop=True), validation.reset_index(drop=True)


def save_eda(train_raw, train_clean, out):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    train_raw["label"].value_counts().sort_values().plot(kind="barh", ax=ax, colormap="viridis")
    ax.set(title="ArPanEmo Training Class Distribution", xlabel="Posts", ylabel="Emotion")
    fig.tight_layout(); fig.savefig(out / "eda_class_distribution.png", dpi=220); plt.close(fig)
    lengths = train_clean["clean_text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(lengths, bins=40, ax=ax)
    ax.axvline(lengths.median(), color="red", linestyle="--", label=f"Median={lengths.median():.0f}")
    ax.set(title="Cleaned Text Lengths", xlabel="Words", ylabel="Posts"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "eda_text_lengths.png", dpi=220); plt.close(fig)


def load_or_prepare(args, out):
    raw_train, raw_test = Path(args.train_raw), Path(args.test_raw)
    data = Path(args.data_dir)
    if raw_train.exists() and raw_test.exists():
        train_raw = pd.read_csv(raw_train, encoding="utf-8-sig")
        test_raw = pd.read_csv(raw_test, encoding="utf-8-sig")
        description = {"train": inspect(train_raw, "train"), "test": inspect(test_raw, "test")}
        train_clean, train_log = preprocess(train_raw)
        test_clean, test_log = preprocess(test_raw)
        overlap = set(train_clean.clean_text) & set(test_clean.clean_text)
        overlap_rows = int(test_clean.clean_text.isin(overlap).sum())
        test_clean = test_clean[~test_clean.clean_text.isin(overlap)].reset_index(drop=True)
        test_log["removed_train_test_overlap"] = overlap_rows
        train_split, validation = split_clean_train(train_clean)
        save_eda(train_raw, train_clean, out)
    else:
        names = {
            "train_clean": "ArPanEmo_train_clean.csv", "test_clean": "ArPanEmo_test_clean.csv",
            "train_split": "ArPanEmo_train_split.csv", "validation": "ArPanEmo_validation_split.csv",
        }
        frames = {key: pd.read_csv(data / value, encoding="utf-8-sig") for key, value in names.items()}
        train_clean, test_clean = frames["train_clean"], frames["test_clean"]
        train_split, validation = frames["train_split"], frames["validation"]
        description = {"status": "loaded from saved cleaned/split CSV files"}
        train_log = test_log = {"status": "preprocessing already completed"}

    for frame, name in [(train_clean,"train_clean"),(test_clean,"test_clean"),(train_split,"train_split"),(validation,"validation")]:
        if not {"post", "label", "clean_text"}.issubset(frame.columns):
            raise ValueError(f"{name} lacks post, label, or clean_text")
    train_clean.to_csv(out / "ArPanEmo_train_clean.csv", index=False, encoding="utf-8-sig")
    test_clean.to_csv(out / "ArPanEmo_test_clean.csv", index=False, encoding="utf-8-sig")
    train_split.to_csv(out / "ArPanEmo_train_split.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(out / "ArPanEmo_validation_split.csv", index=False, encoding="utf-8-sig")
    return train_clean, test_clean, train_split, validation, description, train_log, test_log


def score(y_true, y_pred):
    represented = sorted(pd.Series(y_true).unique())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true,y_pred,labels=represented,average="macro",zero_division=0)),
        "macro_recall": float(recall_score(y_true,y_pred,labels=represented,average="macro",zero_division=0)),
        "macro_f1": float(f1_score(y_true,y_pred,labels=represented,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
    }


def save_eval(model_name, split_name, frame, predictions, out):
    y = frame["label"]
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / f"{model_name}_{split_name}_report.csv", encoding="utf-8-sig")
    errors = frame[["number","post","clean_text","label"]].copy()
    errors["predicted_label"] = predictions
    errors[errors.label != errors.predicted_label].to_csv(
        out / f"{model_name}_{split_name}_errors.csv", index=False, encoding="utf-8-sig"
    )
    cm = confusion_matrix(y, predictions, labels=LABELS)
    fig, ax = plt.subplots(figsize=(11,9))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=45, cmap="Blues")
    ax.set_title(f"{model_name} - {split_name}")
    fig.tight_layout(); fig.savefig(out / f"{model_name}_{split_name}_confusion.png", dpi=220); plt.close(fig)
    return score(y, predictions)


def run_arabert(train, validation, test, out):
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers, and accelerate to run AraBERT") from exc
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    encoder = LabelEncoder().fit(LABELS)
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    class EmotionDataset(Dataset):
        def __init__(self, frame):
            self.enc = tokenizer(frame.clean_text.tolist(), truncation=True, padding=True, max_length=128)
            self.labels = encoder.transform(frame.label)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long); return item
    train_ds, val_ds, test_ds = EmotionDataset(train), EmotionDataset(validation), EmotionDataset(test)
    model = AutoModelForSequenceClassification.from_pretrained(
        "aubmindlab/bert-base-arabertv02", num_labels=len(LABELS),
        id2label={i:x for i,x in enumerate(encoder.classes_)}, label2id={x:i for i,x in enumerate(encoder.classes_)},
    )
    def hf_metrics(p):
        pred=np.argmax(p.predictions if hasattr(p,"predictions") else p[0],axis=-1)
        true=p.label_ids if hasattr(p,"label_ids") else p[1]
        pr,rc,mf,_=precision_recall_fscore_support(true,pred,average="macro",zero_division=0)
        return {"accuracy":accuracy_score(true,pred),"macro_precision":pr,"macro_recall":rc,"macro_f1":mf,"weighted_f1":f1_score(true,pred,average="weighted",zero_division=0)}
    common=dict(output_dir=str(out/"arabert_checkpoints"),save_strategy="epoch",learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=8,num_train_epochs=3,weight_decay=0.01,load_best_model_at_end=True,metric_for_best_model="macro_f1",report_to="none",seed=SEED,data_seed=SEED,save_total_limit=1)
    try: training_args=TrainingArguments(eval_strategy="epoch",**common)
    except TypeError: training_args=TrainingArguments(evaluation_strategy="epoch",**common)
    trainer=Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=val_ds,compute_metrics=hf_metrics)
    trainer.train()
    results={}
    for name,frame,dataset in [("validation",validation,val_ds),("test",test,test_ds)]:
        output=trainer.predict(dataset); pred=encoder.inverse_transform(np.argmax(output.predictions,axis=-1))
        results[name]=save_eval("arabert",name,frame,pred,out)
    trainer.save_model(str(out/"arabert_best_model")); tokenizer.save_pretrained(str(out/"arabert_best_model"))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-raw",default="ArPanEmo_train.csv")
    parser.add_argument("--test-raw",default="ArPanEmo_test.csv")
    parser.add_argument("--data-dir",default=".")
    parser.add_argument("--output",default="phase2_complete_outputs")
    parser.add_argument("--skip-arabert",action="store_true")
    args=parser.parse_args(); set_seed(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    train_clean,test_clean,train,val,description,train_log,test_log=load_or_prepare(args,out)

    extractor=FeatureUnion([
        ("word",TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)),
        ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True)),
    ])
    x_train=extractor.fit_transform(train.clean_text); x_val=extractor.transform(val.clean_text); x_test=extractor.transform(test_clean.clean_text)
    save_npz(out/"train_tfidf.npz",x_train); save_npz(out/"validation_tfidf.npz",x_val); save_npz(out/"test_tfidf.npz",x_test)
    joblib.dump(extractor,out/"tfidf_feature_extractor.joblib")

    dummy=DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train),1)),train.label)
    baseline_val=dummy.predict(np.zeros((len(val),1))); baseline_test=dummy.predict(np.zeros((len(test_clean),1)))
    lr=LogisticRegression(max_iter=1000,random_state=SEED).fit(x_train,train.label)
    lr_val=lr.predict(x_val); lr_test=lr.predict(x_test)
    joblib.dump({"extractor":extractor,"classifier":lr},out/"tfidf_logistic_regression.joblib")

    summary={
        "dataset_description":description,"preprocessing":{"train":train_log,"test":test_log},
        "rows":{"clean_train":len(train_clean),"train":len(train),"validation":len(val),"test":len(test_clean)},
        "features":{"word":len(extractor.transformer_list[0][1].get_feature_names_out()),"character":len(extractor.transformer_list[1][1].get_feature_names_out()),"combined":x_train.shape[1]},
        "baseline":{"validation":score(val.label,baseline_val),"test":score(test_clean.label,baseline_test)},
        "logistic_regression":{"validation":save_eval("logistic_regression","validation",val,lr_val,out),"test":save_eval("logistic_regression","test",test_clean,lr_test,out)},
    }
    summary["arabert"]={"status":"skipped"} if args.skip_arabert else run_arabert(train,val,test_clean,out)
    with (out/"phase2_complete_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
