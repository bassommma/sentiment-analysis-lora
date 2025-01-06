from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_lin", "k_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)


peftmodel=get_peft_model(model,lora_config)