from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

TRANS_MODEL = "facebook/nllb-200-distilled-600M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL)

model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANS_MODEL
).to(DEVICE)


def translate_text(text, tgt_lang):
    """
    English -> target language
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=256
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def back_translate_nllb(
    text,
    tgt_lang="eng_Latn"
):
    """
    Target language -> English
    """

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=256
        )

        return tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True
        )

    except:
        return ""
