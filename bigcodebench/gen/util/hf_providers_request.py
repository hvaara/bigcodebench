import time

from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import TextGenerationOutput


def make_request(
    client: InferenceClient,
    message: str,
    model: str,
    temperature: float,
    n: int,
    max_new_tokens: int = 2048,
) -> TextGenerationOutput:
    response = client.text_generation(
        model=model,
        inputs=message,
        provider="hf-inference",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    return response


def make_auto_request(*args, **kwargs) -> TextGenerationOutput:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        # except ResourceExhausted as e:
        #     print("Rate limit exceeded. Waiting...", e.message)
        #     time.sleep(10)
        # except GoogleAPICallError as e:
        #     print(e.message)
        #     time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(100)
    return ret