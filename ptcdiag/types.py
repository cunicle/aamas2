from dataclasses import asdict, dataclass, field


@dataclass
class Example:
    """One single-turn tool-calling instance, in BFCL format.

    functions:    BFCL function docs ({"name", "description", "parameters": {"type": "dict", ...}}).
    ground_truth: BFCL possible answers, one dict per expected call:
                  [{func_name: {param: [acceptable values...]}}]. "" in a list means "may be omitted".
    category:     BFCL category name ("parallel", "live_parallel_multiple", ...) or a probe
                  category ("probe_parallel", "probe_parallel_multiple"). The checker dispatches on it.
    """

    id: str
    category: str
    messages: list
    functions: list
    ground_truth: list
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def gold_calls(self):
        """[(func_name, {param: acceptable_list})] in ground-truth order."""
        return [(name, params) for g in self.ground_truth for name, params in g.items()]

    def function(self, name):
        for f in self.functions:
            if f["name"] == name:
                return f
        return None
