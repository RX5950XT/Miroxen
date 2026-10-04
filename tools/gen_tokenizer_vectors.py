#!/usr/bin/env python3
# 用 HuggingFace tokenizers 產生 Qwen2 / MiniMax byte-level BPE 的對照向量。
# 執行：uv run --with tokenizers --python <refenv python> tools/gen_tokenizer_vectors.py
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "tokenizer_vectors.txt"
QWEN = Path(r"D:\Workspace\Personal_Project\Miroxen\models\qwen-image-2.1\hf\processor\tokenizer.json")
MINI = Path(r"D:\Workspace\Personal_Project\Miroxen\models\minimax-h3\hf\tokenizer\tokenizer.json")

SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


def cases():
    seen = set()
    out = []

    def add(s):
        if s not in seen:
            seen.add(s)
            out.append(s)

    add("")
    add(" ")
    add("\n")
    add("\r")
    add("\t")
    add("\r\n")
    add("\n\n")
    add("\t\t")
    add(" \t ")
    add("!")
    add("hello")
    add("Hello")
    add("HELLO")
    add(" hello")
    add("  hello")
    add("   hello")
    add("hello ")
    add("hello  ")
    add("hello   ")
    add("a  b")
    add("a   b")
    add("a\tb")
    add("a\n")
    add("a\n\nb")
    add(" \n ")
    add(" \n\n ")
    add("  \nhello")
    add("  \n  \n  ")
    add("end  \n")
    add("hello\nworld")
    add("hello\r\nworld")
    add("  \r\n  x")
    add("a \n b")
    add("word\n")
    add("word\n\n")
    add("it's")
    add("IT'S")
    add("don't")
    add("I'll")
    add("THEY'RE")
    add("I'm")
    add("WE'VE")
    add("YOU'LL")
    add("HE'D")
    add("'s")
    add("'re")
    add("'ll")
    add("'d")
    add("'m")
    add("'t")
    add("'ve")
    add("'real")
    add("'something")
    add("'\u017fomething")
    add("''s")
    add("rock'n'roll")
    add("l'amour")
    add("123")
    add("0")
    add("3.14159")
    add("1,234")
    add("-5")
    add("a 1 b")
    add("123abc")
    add("abc123")
    add("!!!")
    add(" ... ")
    add("。。。")
    add("a—b")
    add("a\u00a0b")
    add("a\u3000b")
    add("line\u2028next")
    add("line\u2029next")
    add("🙂")
    add("a🙂b")
    add("👍🏽")
    add("👨‍👩‍👧‍👦")
    add("🇨🇳")
    add("\U0001f600")
    add("e\u0301")
    add("E\u0301")
    add("cafe\u0301")
    add("\u212b")
    add("\u0044\u0307\u0323")
    add("\u1100\u1161")
    add("\u1100\u0301\u1161")
    add("\uac00\u11a8")
    add("\uf900")
    add("\u00e9\u0301")
    add("\ufb00")
    add("\u0958")
    add("na\u0303o")
    add("a\u0308\u0301")
    add("１２３")
    add("١٢٣")
    add("a\u200bb")
    add("\u00ad")
    add("\ufe0fa")
    add("你好世界")
    add("你 好")
    add("繁體中文測試：這是一隻貓，請畫出來。")
    add("简体中文测试：请画一只猫。")
    add("日本語のテストです。こんにちは！")
    add("한국어 테스트입니다.")
    add("Qwen-Image 模型可以畫「水墨貓」。")
    add("價格是 $3.50，約 €3。")
    add("مرحبا بالعالم")
    add("שלום עולם")
    add("hello עברית world")
    add("أَ")
    add("a=b+c*d")
    add("https://example.com/a?q=1&x=2")
    add("email me at a.b@example.com")
    add('quote: "hello"')
    add("path C:\\Windows\\Temp")
    add("def f(x):\n    return x + 1\n")
    add("int main() {\n  return 0;\n}\n")
    add('{"a": [1, 2]}\n')
    add("tabs\there\tand\tthere")
    add("line1\r\nline2\r\n")
    add("<|im_start")
    add("<|im_start|>hi<|im_end|>")
    add("foo <|im_start|>\nbar")
    add("<think>\nhello\n</think>")
    add("foo<|im_start|>bar")
    add("<|image_pad|><|image_pad|>")
    add("<think></think>")
    chat = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "畫一張貓，Traditional 與 简体 混排。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    add(chat)
    add(
        "The quick brown fox jumps over 123 lazy dogs.\n"
        "你好世界，這是繁體中文與简体中文。\n"
        "日本語と한국어 🙂 rock'n'roll it's 3.14\n"
        "<|im_start|>user\n畫一隻貓<|im_end|>\n"
    )
    for sp in SPECIALS:
        add(sp)
        add("x" + sp + "y")
        add(" " + sp + "\n")
    add("".join(SPECIALS))
    words = ["cat", "畫", "猫", "ネコ", "고양이", "123", "code", "🙂", "Å", "e\u0301"]
    for w in words:
        add(w + " " + w)
        add(w + "\n" + w)
        add("<|im_start|>" + w + "<|im_end|>")
        add(w + "<think>" + w)
    # 較長的英文，讓 BPE 合併跨過很多 pair。
    add(
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
        "Sphinx of black quartz, judge my vow."
    )
    add("   \n\t  \r\n\t\thello   world\t\n")
    add("()[]{}<>")
    add("...")
    add("——")
    add("「」『』")
    add("…")
    add("™")
    add("①②③")
    add("½")
    add("ß")
    add("İ")
    add("i\u0307")
    add("ﬁ")
    add("é")
    add("ñ")
    add("한글과 漢字 と English")
    add("user: draw a cat\nassistant: ok")
    add(" \u3000你好")
    add("🙂🙂🙂")
    add("a\u00a0\u00a0b")
    add("'S 'T 'RE 'VE 'M 'LL 'D")
    add("don't stop")
    add("end")
    if len(out) < 200:
        raise SystemExit(f"只有 {len(out)} 筆，不足 200")
    return out


def write_section(fh, name, tk, samples):
    fh.write(f"[{name}]\n")
    notes = 0
    for text in samples:
        ids = tk.encode(text).ids
        dec = tk.decode(ids, skip_special_tokens=False)
        nfc = unicodedata.normalize("NFC", text)
        if dec != nfc:
            notes += 1
            if notes <= 5:
                print(f"NOTE {name} decode != NFC {text!r} dec={dec!r} nfc={nfc!r}")
        fh.write("T " + text.encode("utf-8").hex() + "\n")
        fh.write("D " + dec.encode("utf-8").hex() + "\n")
        fh.write("I " + " ".join(str(i) for i in ids) + "\n")
    return notes


def main():
    if not QWEN.is_file():
        raise SystemExit(f"找不到 Qwen tokenizer：{QWEN}")
    samples = cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    qwen = Tokenizer.from_file(str(QWEN))
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("MXTOK1\n")
        fh.write("# generated by tools/gen_tokenizer_vectors.py — do not edit by hand\n")
        n1 = write_section(fh, "qwen", qwen, samples)
        n2 = 0
        if MINI.is_file():
            mini = Tokenizer.from_file(str(MINI))
            n2 = write_section(fh, "minimax", mini, samples)
            differ = 0
            for text in samples:
                if qwen.encode(text).ids != mini.encode(text).ids:
                    differ += 1
            print(f"qwen vs minimax id mismatches: {differ}")
        else:
            print(f"skip minimax, missing {MINI}")
    size = OUT.stat().st_size
    print(f"wrote {OUT} cases={len(samples)} bytes={size} nfc_notes={n1}+{n2}")
    if size >= 500 * 1024:
        raise SystemExit("向量檔超過 500KB")


if __name__ == "__main__":
    main()
