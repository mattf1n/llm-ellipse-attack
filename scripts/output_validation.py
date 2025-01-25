from transformers import AutoModelForCausalLM, AutoTokenizer
import fire

def main(
        model_name: str = "EleutherAI/pythia-70m", 
        prompt: str = "Here are some inputs",
        ):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(prompt, return_tensors="pt")
    print(model)
    ellipse = model

if __name__ == "__main__":
    fire.Fire(main)
