import sys
from contextlib import contextmanager

LINE = "─" * 64
LINE_EQ = "═" * 64


@contextmanager
def suppress_stdout():
    import io
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _title(text):
    print(f"\n  [{text}]")


def _kv(key, value, indent=4, total_width=40):
    prefix = " " * indent
    label = f"{key} "
    pad_len = max(2, total_width - len(label))
    dots = " " + "." * pad_len + " "
    print(f"{prefix}{label}{dots} {value}")


def _bool_en(b):
    return "yes" if b else "no"