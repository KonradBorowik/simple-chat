from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def main():
    while True:
        conversation_history = []
        history_string = "\n".join(conversation_history)
        input_text = input("> ")
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
        # print(inputs)
        outputs = model.generate(**inputs)
        # print(outputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        conversation_history.append(input_text)
        conversation_history.append(response)


if __name__ == "__main__":
    main()
