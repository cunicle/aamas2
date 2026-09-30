"""Prompt construction and output parsing shared by dLLMs and AR baselines.

Every model sees the same system prompt (BFCL "prompt mode"): the function docs as
JSON, and an instruction to answer with a JSON array of calls. Output format:

    [{"name": "<function>", "arguments": {"<param>": <value>, ...}}, ...]

The parser records character spans for every value, so an error found in a parsed
call can be mapped back to the canvas positions (and hence decoding steps) that
produced it.
"""

import json
from dataclasses import dataclass, field

SYSTEM_TEMPLATE = (
    "You are an expert in composing function calls. You are given a question and a set of "
    "possible functions. Based on the question, make one or more function calls to achieve "
    "the purpose.\n\n"
    "Respond ONLY with a JSON array of function calls, one object per call, in the form\n"
    '[{{"name": "<function name>", "arguments": {{"<param>": <value>, ...}}}}]\n'
    "If the question needs several calls, include all of them in the array. "
    "Do not add any other text.\n\n"
    "Here is a list of functions in JSON format that you can invoke:\n{functions}\n"
)


def system_prompt(functions):
    return SYSTEM_TEMPLATE.format(functions=json.dumps(functions, ensure_ascii=False, indent=2))


def build_messages(example):
    """System message with the tools, followed by the example's own messages.

    Some BFCL live entries already contain a system message; the tool prompt is
    prepended to it rather than added as a second system message.
    """
    sys_text = system_prompt(example.functions)
    msgs = [dict(m) for m in example.messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = sys_text + "\n" + msgs[0]["content"]
        return msgs
    return [{"role": "system", "content": sys_text}] + msgs


def render_prompt(tokenizer, example):
    """Chat-templated prompt string ending with the assistant generation prefix."""
    msgs = build_messages(example)
    try:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        # Templates without a system role: fold the system text into the first user turn.
        sys_text = msgs[0]["content"]
        rest = [dict(m) for m in msgs[1:]]
        rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
        return tokenizer.apply_chat_template(rest, add_generation_prompt=True, tokenize=False)


# ---------------------------------------------------------------- JSON with spans


@dataclass
class Node:
    value: object
    start: int
    end: int  # exclusive
    children: object = None  # dict[str, Node] for objects, list[Node] for arrays


class _Parser:
    def __init__(self, s):
        self.s = s

    def ws(self, i):
        while i < len(self.s) and self.s[i] in " \t\r\n":
            i += 1
        return i

    def parse(self, i):
        i = self.ws(i)
        if i >= len(self.s):
            raise ValueError("unexpected end")
        c = self.s[i]
        if c == "{":
            return self.obj(i)
        if c == "[":
            return self.arr(i)
        if c == '"':
            return self.string(i)
        return self.scalar(i)

    def obj(self, i):
        start, i = i, self.ws(i + 1)
        vals, kids = {}, {}
        if self.s[i:i + 1] == "}":
            return Node(vals, start, i + 1, kids)
        while True:
            key = self.string(self.ws(i))
            i = self.ws(key.end)
            if self.s[i:i + 1] != ":":
                raise ValueError(f"expected ':' at {i}")
            child = self.parse(i + 1)
            vals[key.value], kids[key.value] = child.value, child
            i = self.ws(child.end)
            if self.s[i:i + 1] == ",":
                i += 1
                continue
            if self.s[i:i + 1] == "}":
                return Node(vals, start, i + 1, kids)
            raise ValueError(f"expected ',' or '}}' at {i}")

    def arr(self, i):
        start, i = i, self.ws(i + 1)
        vals, kids = [], []
        if self.s[i:i + 1] == "]":
            return Node(vals, start, i + 1, kids)
        while True:
            child = self.parse(i)
            vals.append(child.value)
            kids.append(child)
            i = self.ws(child.end)
            if self.s[i:i + 1] == ",":
                i += 1
                continue
            if self.s[i:i + 1] == "]":
                return Node(vals, start, i + 1, kids)
            raise ValueError(f"expected ',' or ']' at {i}")

    def string(self, i):
        if self.s[i:i + 1] != '"':
            raise ValueError(f"expected string at {i}")
        value, end = json.decoder.scanstring(self.s, i + 1)
        return Node(value, i, end)

    def scalar(self, i):
        for lit, val in (("true", True), ("false", False), ("null", None)):
            if self.s.startswith(lit, i):
                return Node(val, i, i + len(lit))
        j = i
        while j < len(self.s) and self.s[j] in "+-0123456789.eE":
            j += 1
        if j == i:
            raise ValueError(f"unexpected character {self.s[i]!r} at {i}")
        return Node(json.loads(self.s[i:j]), i, j)


def parse_json_with_spans(s, start=0):
    return _Parser(s).parse(start)


# ---------------------------------------------------------------- tool-call extraction


@dataclass
class ParsedOutput:
    text: str
    syntax_ok: bool
    calls: list = field(default_factory=list)  # [{"name": str, "arguments": dict}]
    error: str = ""
    root: Node = None  # span tree of the JSON array, when syntax_ok

    def model_output(self):
        """BFCL checker format: [{name: arguments}]."""
        return [{c["name"]: c["arguments"]} for c in self.calls]

    def value_span(self, call_idx, param):
        """Character span of `arguments[param]` in call `call_idx`, or None."""
        if self.root is None:
            return None
        call = self.root.children[call_idx]
        args = call.children.get("arguments") if call.children else None
        if args is None or args.children is None or param not in args.children:
            return None
        n = args.children[param]
        return n.start, n.end

    def call_span(self, call_idx):
        n = self.root.children[call_idx]
        return n.start, n.end


def parse_tool_calls(text):
    """Parse the first JSON array in `text` as a list of tool calls.

    syntax_ok means: a JSON array whose items are objects with a string "name" and
    an object "arguments". Anything after the array (e.g. padding) is ignored.
    """
    start = text.find("[")
    if start < 0:
        return ParsedOutput(text, False, error="no '[' found")
    try:
        root = parse_json_with_spans(text, start)
    except (ValueError, IndexError, json.JSONDecodeError) as e:
        return ParsedOutput(text, False, error=f"json: {e}")
    if not isinstance(root.value, list):
        return ParsedOutput(text, False, error="top level is not an array")
    calls = []
    for item in root.value:
        if not (isinstance(item, dict) and isinstance(item.get("name"), str)
                and isinstance(item.get("arguments"), dict)):
            return ParsedOutput(text, False, error=f"malformed call: {item!r}")
        calls.append({"name": item["name"], "arguments": item["arguments"]})
    return ParsedOutput(text, True, calls=calls, root=root)


def token_char_offsets(tokenizer, ids):
    """Character end offset of each token in `tokenizer.decode(ids)`.

    Uses incremental decoding, so it is exact for byte-level BPE tokenizers where
    individual tokens may not decode to valid text on their own.
    """
    ends, prev = [], 0
    for i in range(len(ids)):
        n = len(tokenizer.decode(ids[:i + 1], skip_special_tokens=False))
        ends.append(max(n, prev))
        prev = ends[-1]
    return ends


def tokens_in_span(offset_ends, span):
    """Indices of tokens whose decoded characters overlap the half-open span."""
    lo, hi = span
    out, prev = [], 0
    for i, end in enumerate(offset_ends):
        if end > lo and prev < hi:
            out.append(i)
        prev = end
    return out
