import torch
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

checkpoint = "Mia2024/CS5100TextSummarization"
checkpoint = "facebook/bart-large-cnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, st_container, st_info_container, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.st_container = st_container
        self.st_info_container = st_info_container
        self.text = ""
        self.start_time = None
        self.first_token_time = None
        self.total_tokens = 0

    def on_finalized_text(self, text: str, stream_end: bool=False):
        if self.start_time is None:
            self.start_time = time.time()

        if self.first_token_time is None and len(text.strip()) > 0:
            self.first_token_time = time.time()

        self.text += text

        self.total_tokens += len(text.split())
        self.st_container.markdown("###### " + self.text)
        time.sleep(0.03)


def generate_summary(input_text, st_container, st_info_container) -> str:
    generation_config = GenerationConfig(
            min_new_tokens=10,
            max_new_tokens=256,
            temperature=0.9,
            top_p=1.0,
            top_k=50         
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    prefix = "Summarize the following conversation: \n###\n"
    suffix = "\n### Summary:"
    target_length = max(1, int(0.15 * len(input_text.split())))

    input_ids = tokenizer.encode(prefix + input_text + f"The generated summary should be around {target_length} words." + suffix, return_tensors="pt")

    # Initialize the Streamlit container and streamer
    streamer = StreamlitTextStreamer(tokenizer, st_container, st_info_container, skip_special_tokens=True, decoder_start_token_id=3)

    model.generate(input_ids, streamer=streamer, do_sample=True, generation_config=generation_config)
    

    