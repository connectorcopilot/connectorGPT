import os

from transformers import (
    CodeGenTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def textFile(dataText, name):
    all_text = ""
    for text in dataText:
        # print(description)
        all_text = all_text + text + '<|endoftext|> \n '
    with open(
            f'./content/sample_data/{name}.txt', 'w',
            encoding="utf-8") as f:
        # print(test)
        # print(f"lol", f.write(' <|startoftext|> '+ '[DES]'+ description +'[TIT]' + title+'.' + " <|endoftext|> "))
        print(f"{all_text}", file=f)

    # Load the tokenizer and pre-trained model


# tokenizer = AutoTokenizer.from_pretrained("salesforce/codegen-350M-multi")
tokenizer = CodeGenTokenizer.from_pretrained("salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("salesforce/codegen-350M-multi")

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model.resize_token_embeddings(len(tokenizer))

directory = "./connector"
text_files = []

# Convert .go files in the "connector" directory to text files
for file in os.listdir(directory):
    if file.endswith(".go"):
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as f:
            text = f.read()
            text_files.append(text)

# Split the data into training, validation, and test sets
train_size = int(0.8 * len(text_files))
train_texts = text_files[:train_size]
val_texts = text_files[train_size:]

textFile(train_texts, "train")
textFile(val_texts, "val")

# Convert the data to a dataset format
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='./content/sample_data/train.txt',
    block_size=128)

val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='./content/sample_data/val.txt',
    block_size=128)

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
)

# Create a Trainer instance and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./codegen-multi-350M")
