from transformers import CodeGenTokenizer, AutoModelForCausalLM
tokenizer = CodeGenTokenizer.from_pretrained("./codegen-multi-350M/", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./codegen-multi-350M/")

inputs = "REST API for /v1/testconnection"

completion = model.generate(**tokenizer(inputs, return_tensors="pt"))

print(tokenizer.decode(completion[0]))
