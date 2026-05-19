def generate_phi4_native_answer(
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
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        gen_ids,
        skip_special_tokens=True
    ).strip()
