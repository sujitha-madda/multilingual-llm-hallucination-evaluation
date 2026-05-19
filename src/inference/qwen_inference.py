def generate_qwen_native_answer(
    tokenizer,
    model,
    question,
    language="Hindi"
):

    prompt = (
        f"You are a knowledgeable assistant.\n"
        f"Answer clearly and concisely in {language} only.\n"
        f"Do not repeat the question.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        gen_ids,
        skip_special_tokens=True
    ).strip()
