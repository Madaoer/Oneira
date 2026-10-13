"""Command-line arguments on top of a yaml config: command line > yaml > defaults."""

import yaml


def parse(parser, defaults):
    """Options in `defaults` are added with default None so that yaml values can fill them in."""
    for key in defaults:
        parser.add_argument(
            "--" + key.replace("_", "-"), default=None, type=type(defaults[key]) if defaults[key] is not None else str
        )
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config)) if getattr(args, "config", None) else {}
    for key, value in {**defaults, **(cfg or {})}.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args
