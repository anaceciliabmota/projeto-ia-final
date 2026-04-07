import os
from pathlib import Path

import pandas as pd
import requests
import torch
from datasets import Dataset
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_CHECKPOINT = "Salesforce/blip-image-captioning-base"
POSTERS_DIR = Path("data/posters")
MAX_LEN = 256


def carregar_dados(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().dropna(subset=["sinopse"]).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def download_poster(url: str) -> str:
    POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    path = POSTERS_DIR / url.split("/")[-1]
    if not path.exists():
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        path.write_bytes(r.content)
    return str(path)


def add_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    paths = []
    for url in df["poster"]:
        try:
            paths.append(download_poster(url))
        except Exception:
            paths.append(None)
    df = df.copy()
    df["image_path"] = paths
    return df.dropna(subset=["image_path"]).reset_index(drop=True)


def build_preprocess_fn(processor: BlipProcessor):
    def preprocess(example: dict) -> dict:
        image = PILImage.open(example["image_path"]).convert("RGB")
        prompt = "Gere uma sinopse de terror para o filme: " + example["titulo"]
        full_text = prompt + " " + example["sinopse"]

        inputs = processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        labels = inputs.input_ids.clone()
        enc_text = processor.tokenizer(
            full_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        prompt_end_char = len(prompt) + 1
        for i, (_start, end) in enumerate(enc_text.offset_mapping[0]):
            if end == 0 or end <= prompt_end_char:
                labels[0, i] = -100
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }
    return preprocess


def build_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, processor: BlipProcessor):
    cols = ["image_path", "titulo", "sinopse"]
    preprocess = build_preprocess_fn(processor)

    tokenized_train = Dataset.from_pandas(train_df[cols]).map(
        preprocess, batched=False, remove_columns=cols
    )
    tokenized_test = Dataset.from_pandas(test_df[cols]).map(
        preprocess, batched=False, remove_columns=cols
    )
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")
    return tokenized_train, tokenized_test


def carregar_modelo() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    processor = BlipProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    return processor, model


def congelar_camadas(model: BlipForConditionalGeneration) -> BlipForConditionalGeneration:
    for name, param in model.named_parameters():
        if "vision_model.encoder.layers.10" in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {trainable:,} / {total:,}")
    return model


def treinar(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    tokenized_train: Dataset,
    tokenized_test: Dataset,
) -> Seq2SeqTrainer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Treinando em: {device}")

    training_args = Seq2SeqTrainingArguments(
        output_dir="blip-horror",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=processor.tokenizer,
    )
    trainer.train()
    return trainer


def gerar_sinopse(row: pd.Series, model: BlipForConditionalGeneration, processor: BlipProcessor) -> str:
    image = PILImage.open(row["image_path"]).convert("RGB")
    inputs = processor(
        images=image,
        text="Gere uma sinopse de terror para o filme: " + row["titulo"],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    return processor.decode(output[0], skip_special_tokens=True)


def gerar_resultados(
    test_df: pd.DataFrame,
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    output_path: str = "data/resultados.csv",
) -> pd.DataFrame:
    resultados = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Gerando sinopses"):
        try:
            sinopse_gerada = gerar_sinopse(row, model, processor)
        except Exception as e:
            sinopse_gerada = str(e)
        resultados.append({"titulo": row["titulo"], "sinopse_gerada": sinopse_gerada})

    df_resultados = pd.DataFrame(resultados)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(output_path, index=False)
    print(f"Resultados salvos em {output_path}")
    return df_resultados


def main():
    print("Carregando dados...")
    train_df, test_df = carregar_dados("filmes.csv")

    print("Baixando pôsteres...")
    train_df = add_image_paths(train_df)
    test_df = add_image_paths(test_df)

    print("Carregando modelo...")
    processor, model = carregar_modelo()
    model = congelar_camadas(model)

    print("Pré-processando datasets...")
    tokenized_train, tokenized_test = build_datasets(train_df, test_df, processor)

    print("Treinando...")
    treinar(model, processor, tokenized_train, tokenized_test)

    print("Gerando resultados do teste...")
    gerar_resultados(test_df, model, processor, output_path="data/resultados.csv")


if __name__ == "__main__":
    main()
