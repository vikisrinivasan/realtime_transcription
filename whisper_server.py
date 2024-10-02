import torch
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType
import optuna
from optuna.trial import TrialState

# Load pre-trained model and processor
model_name = "openai/whisper-small"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")

# Load IMDA-STT dataset
dataset = load_dataset("mesolitica/IMDA-STT")
dataset = dataset["train"].cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Split dataset into train and validation
dataset = dataset.train_test_split(test_size=0.1)

def objective(trial):
    # Define LoRA Config
    lora_config = LoraConfig(
        r=trial.suggest_categorical("r", [8, 16, 32, 64]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [16, 32, 64, 128]),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=trial.suggest_categorical("lora_dropout", [0.05, 0.1, 0.15]),
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # Load model and apply LoRA
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-singlish-lora-trial-{trial.number}",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        warmup_steps=500,
        max_steps=5000,  # Reduced for faster trials
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Define trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=lambda x: {"input_features": torch.stack([i["input_features"] for i in x]),
                                 "labels": torch.nn.utils.rnn.pad_sequence([torch.LongTensor(i["labels"]) for i in x], 
                                                                            batch_first=True, 
                                                                            padding_value=-100)},
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate()

    return eval_result["eval_wer"]

# Run the hyperparameter search
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train the final model with the best hyperparameters
best_lora_config = LoraConfig(
    r=trial.params["r"],
    lora_alpha=trial.params["lora_alpha"],
    target_modules=["q_proj", "v_proj"],
    lora_dropout=trial.params["lora_dropout"],
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

final_model = WhisperForConditionalGeneration.from_pretrained(model_name)
final_model = get_peft_model(final_model, best_lora_config)

final_training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-singlish-lora-final",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=trial.params["learning_rate"],
    warmup_steps=500,
    max_steps=10000,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

final_trainer = Seq2SeqTrainer(
    args=final_training_args,
    model=final_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=lambda x: {"input_features": torch.stack([i["input_features"] for i in x]),
                             "labels": torch.nn.utils.rnn.pad_sequence([torch.LongTensor(i["labels"]) for i in x], 
                                                                        batch_first=True, 
                                                                        padding_value=-100)},
)

# Train the final model
final_trainer.train()

# Save the final fine-tuned model
final_model.save_pretrained("./whisper-singlish-lora-final")
processor.save_pretrained("./whisper-singlish-lora-final")
