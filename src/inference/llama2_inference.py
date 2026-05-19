def generate_llama_answer(
    tokenizer,
    model,
    question
):

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        gen_ids,
        skip_special_tokens=True
    ).strip()
