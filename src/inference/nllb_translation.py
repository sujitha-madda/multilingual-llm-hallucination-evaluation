def en_to_target(text):
    inputs = trans_tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to("cuda")

    outputs = trans_model.generate(
        **inputs,
        forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids(CONFIG["lang_code"]),
        max_length=256
    )

    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
