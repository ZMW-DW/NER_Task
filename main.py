import torch
from transformers import BertTokenizer, BertModel

def main():
    device = "mps"
    tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-uncased')
    model = BertModel.from_pretrained("./bert-base-multilingual-uncased").to(device)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    print(encoded_input)
    print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]))
    print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0][1:-1]))
    output = model(**encoded_input)
    last_hidden_state: torch.Tensor = output.last_hidden_state
    print(last_hidden_state.shape)

if __name__ == "__main__":
    main()