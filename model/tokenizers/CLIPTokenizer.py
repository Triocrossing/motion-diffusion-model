from typing import Any, Union, List
from clip.clip import _Tokenizer
from clip.simple_tokenizer import whitespace_clean, basic_clean
import torch
from pkg_resources import packaging
import regex as re


class CLIPTokenizer(_Tokenizer):
    def __init__(self):

        super().__init__()

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, placeholder_token):
        tokenizer = CLIPTokenizer()
        old = len(tokenizer.encoder)
        tokenizer.encoder[placeholder_token] = tokenizer.encoder["<|endoftext|>"] + 1
        tokenizer.decoder = {v: k for k, v in tokenizer.encoder.items()}
        return tokenizer

    # def add_tokens(self, placeholder_token):

    #     return len(self.encoder) - old

    # def encode(self, text):
    #     bpe_tokens = []
    #     breakpoint()
    #     text = whitespace_clean(basic_clean(text)).lower()
    #     for token in [text]:
    #         token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
    #         bpe_tokens.append(self.encoder[token])
    #     return bpe_tokens

    def encode(self, text, add_special_tokens=False):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        pre = ""
        post = ""
        if "." in text:
            tokens = re.findall(self.pat, text.split(".")[0]) + ["."]
        else:
            tokens = re.findall(self.pat, text)
        for token in tokens:

            if token == "<":
                pre = "<"
                post = ">"
                continue
            if token == ">":
                pre = ""
                post = ""
                continue

            token = (
                pre
                + "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                + post
            )
            if "<" in token:
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in (token).split(" ")
                )
            # else:
            # if '<' in token:
            #     bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in token.strip('.').split(' '))
            else:
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
                )
        return bpe_tokens

    def convert_tokens_to_ids(self, placeholder_token):
        print(f"new token id: {self.encoder[placeholder_token]}")
        return self.encoder[placeholder_token]

    def __len__(self):
        return len(self.encoder)


# _tokenizer = CLIPTokenizer.from_pretrained('')
# placeholder_token = '<hey>'
# initializer_token = 'hey'
# num_added_tokens = tokenizer.add_tokens(placeholder_token)
# token_ids = [tokenizer.encoder[initializer_token]]
# initializer_token_id = token_ids[0]
# placeholder_token_id = tokenizer.encoder[placeholder_token]

# print(initializer_token_id)
# print(placeholder_token_id)
# k = CLIPTokenizer.from_pretrained()
# breakpoint()
# k.add_tokens('<hey>')

# texts = ['<hey>']
# if isinstance(texts, str):
#     texts = [texts]

# # sot_token = k.encoder["<|startoftext|>"]
# # eot_token = k.encoder["<|endoftext|>"]
# all_tokens = [ k.encode(text) for text in texts]
# print(all_tokens)
# # print(tokenizer.encode([['hey']]))
def tokenize(
    _tokenizer,
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]
    # ks = []
    # for k,v in _tokenizer.byte_encoder.items():
    #     if v=='<' or v=='>':
    #         ks.append(k)
    # for k in ks:
    #     _tokenizer.byte_encoder.pop(k)
    # ks = []
    # for v,k in _tokenizer.byte_decoder.items():
    #     if v=='<' or v=='>':
    #         ks.append(v)
    # for k in ks:
    #     _tokenizer.byte_decoder.pop(k)

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
