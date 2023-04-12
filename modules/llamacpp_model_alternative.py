'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

from llama_cpp import Llama

from modules import shared
from modules.callbacks import Iteratorize

class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    @classmethod
    def from_pretrained(self, path):
        result = self()

        params = {
            'model_path': str(path),
            'n_ctx': 2048,
            'n_parts': 1,
            'n_batch': 2048,
            'seed': 0,
            'n_threads': shared.args.threads or None
        }
        self.model = Llama(**params)

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def generate(self, context="", token_count=20, temperature=0.1, top_p=0.1, top_k=40, repetition_penalty=1.176, callback=None, stop=['\n', '### Human:']):
        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []
        
        if type(context) is str:
            context = context.encode()
        tokens = self.model.tokenize(context)

        output = b""
        count = 0
        for token in self.model.generate(tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repetition_penalty):
            text = self.model.detokenize([token])
            
            any_stop = [s for s in stop_sequences if s in text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = text[: text.index(first_stop)]
                finish_reason = "stop"
                break
            
            output += text
            if callback:
                callback(text.decode())

            count += 1
            if count >= token_count or (token == self.model.token_eos()):
                break

        return output.decode()

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
