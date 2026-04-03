---
name: BLIP Horror Synopsis Fine-Tuning
overview: Criar um notebook de fine-tuning do BLIP (VLM) para gerar sinopses de terror a partir de pôsteres/obras de arte, espelhando a estrutura didática do notebook BERT existente com analogias e explicações conceituais.
todos:
  - id: dataset-build
    content: "Seção 2: construir dataset baixando pôsteres do TMDB e montando HuggingFace Dataset com colunas image/title/synopsis"
    status: pending
  - id: model-load
    content: "Seção 3: carregar BlipProcessor e BlipForConditionalGeneration, mostrar arquitetura e analogia com BERT"
    status: pending
  - id: preprocess
    content: "Seção 4: implementar função de pré-processamento com processor, criar prompt template e campos labels"
    status: pending
  - id: metrics
    content: "Seção 5: implementar compute_metrics com ROUGE via evaluate library"
    status: pending
  - id: train-full
    content: "Seção 6: configurar Seq2SeqTrainingArguments e Seq2SeqTrainer, rodar treinamento completo"
    status: pending
  - id: eval-examples
    content: "Seção 7: avaliar com ROUGE e mostrar exemplos visuais de geração"
    status: pending
  - id: freeze-experiments
    content: "Seções 8-9: experimentos de congelamento (full / decoder-only / parcial) com comparação de ROUGE"
    status: pending
  - id: exercises
    content: "Seção 10: tarefas do exercício incluindo teste com obra de arte real"
    status: pending
isProject: false
---

# Notebook: Fine-Tuning BLIP para Geração de Sinopses de Terror

## Analogia Central: BERT vs BLIP

```
BERT notebook                      BLIP notebook (nosso)
─────────────────────────────────────────────────────────
Entrada: texto (review)        →   Entrada: imagem (pôster) + texto (título)
Tarefa: classificação           →   Tarefa: geração de texto (sinopse)
Dataset: Rotten Tomatoes        →   Dataset: filmes de terror (TMDB)
Modelo: encoder-only            →   Modelo: encoder-decoder multimodal
Métrica: F1                     →   Métrica: ROUGE / BLEU
```

## Estrutura do Notebook (célula a célula)

### Seção 1 — Contexto Conceitual *(análogo à introdução do BERT)*

- O que é um VLM? Diferença para um LLM puro (como BERT).
- Arquitetura BLIP: **Vision Encoder (ViT)** processa a imagem → **Language Decoder (BERT/GPT-like)** gera texto condicionado na imagem. É como juntar "olhos" (ViT) com "boca" (decoder de linguagem).
- Analogia direta: no notebook BERT, o encoder lia palavras e uma cabeça de classificação respondia "positivo/negativo". Aqui o ViT "lê" a imagem e o decoder "responde" com a sinopse.

### Seção 2 — Construção do Dataset *(análogo a `Importando dataset`)*

Arquivo de referência: `[getting_database.py](getting_database.py)` já busca filmes via TMDB.

O notebook irá:

1. Chamar `buscar_filmes_terror()` para obter lista de dicts `{titulo, sinopse, poster_url}`.
2. Baixar cada imagem de pôster com `requests` → salvar localmente.
3. Montar um `datasets.Dataset` do HuggingFace com colunas `image`, `title`, `synopsis`.
4. Fazer split train/test (ex.: 80/20) — mesmo padrão do `rotten_tomatoes["train"]`/`["test"]`.

> Conceito: no BERT o texto "já era texto". Aqui a imagem precisa ser baixada e depois convertida pelo `BlipProcessor` em tensores de pixels normalizados (como tokenizar, mas para pixels).

### Seção 3 — Carregando o Modelo BLIP *(análogo a `Carregando o modelo BERT`)*

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_CHECKPOINT = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_CHECKPOINT)
model = BlipForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
```

- Mostrar a arquitetura: `model.vision_model` (ViT) + `model.text_decoder`.
- Analogia: `AutoModelForSequenceClassification` tinha `bert.*` + `classifier.*`. Aqui temos `vision_model.*` + `text_decoder.*`.

### Seção 4 — Pré-processamento *(análogo a `Tokenizando a entrada`)*

Criar um `Dataset.map()` que aplica o `BlipProcessor`:

```python
def preprocess(batch):
    inputs = processor(
        images=batch["image"],
        text=["Generate a horror synopsis: " + t for t in batch["title"]],
        return_tensors="pt", padding=True, truncation=True
    )
    inputs["labels"] = processor.tokenizer(
        batch["synopsis"], return_tensors="pt", padding=True, truncation=True
    ).input_ids
    return inputs
```

- Aqui o "prompt" `"Generate a horror synopsis: <título>"` é o equivalente do campo `text` do Rotten Tomatoes — é a entrada linguística que guia a geração.
- O campo `labels` são os tokens da sinopse — o que o modelo deve aprender a gerar (equivalente ao `label` 0/1 do BERT, mas é uma sequência inteira).

### Seção 5 — Definição de Métricas *(análogo a `Definição de métricas`)*

- No BERT: F1 (mede acerto em classificação binária).
- Aqui: **ROUGE-L** (mede sobreposição de sequências entre sinopse gerada e real) via `evaluate.load("rouge")`.
- Explicação curta: ROUGE-L de 0.3 significa que ~30% das sequências da sinopse real aparecem no texto gerado.

### Seção 6 — Treinamento Completo *(análogo a `Treinamento`)*

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="blip-horror",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    predict_with_generate=True,
    save_strategy="epoch",
    report_to="none",
)
trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
trainer.train()
```

- Usa `Seq2SeqTrainer` em vez de `Trainer` porque a tarefa é geração (seq2seq), não classificação.
- Loss: **cross-entropy sobre os tokens da sinopse** — o modelo aprende a prever a próxima palavra da sinopse dado a imagem + título. Análogo ao cross-entropy do BERT, mas sobre uma sequência e não um único label.

### Seção 7 — Avaliação e Geração de Exemplos *(análogo a `Avaliando os resultados`)*

- `trainer.evaluate()` com ROUGE.
- Gerar sinopse para uma imagem de teste com `model.generate(...)` e mostrar o resultado visualmente (imagem + sinopse gerada).

### Seção 8 — O que foi treinado? *(análogo a `...mas o que foi treinado afinal?`)*

- Imprimir `requires_grad` de cada componente: `vision_model`, `text_decoder`.
- Mostrar que ambos foram treinados no fine-tune completo.

### Seção 9 — Congelamento de Camadas *(análogo a `Congelamento de Camadas`)*

Três experimentos espelhando o notebook BERT:


| Experimento    | O que congela                              | Treina                  |
| -------------- | ------------------------------------------ | ----------------------- |
| Full fine-tune | nada                                       | tudo                    |
| Só decoder     | `vision_model.`*                           | `text_decoder.`*        |
| Parcial        | `vision_model` + primeiras camadas decoder | últimas camadas decoder |


Conceito: o ViT já "sabe ver" de imagens gerais (ImageNet). Congelar o encoder e treinar só o decoder é mais rápido e usa menos GPU — boa estratégia quando o dataset é pequeno.

### Seção 10 — Tarefas do Exercício *(análogo a `Tarefas do Exercício`)*

1. Comparar ROUGE entre full fine-tune vs. decoder-only.
2. Congelar as N primeiras camadas do decoder e avaliar.
3. Testar com uma obra de arte real (ex.: "A Noite Estrelada" de Van Gogh) — entrada nunca vista.

## Arquivos e dependências

- `[getting_database.py](getting_database.py)` — reutilizado para montar o dataset
- Dependências novas: `transformers>=4.35`, `datasets`, `evaluate`, `Pillow`, `rouge_score`
- Recomendado: rodar no Google Colab com GPU T4 (mesmo ambiente do notebook BERT)

