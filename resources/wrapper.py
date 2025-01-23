class Wrapper:
    def __init__(self, wrap_rule, exceptions={}):
        self.wrap_rule = wrap_rule
        self.exceptions = exceptions

    def wrap_model(self, current_module, prefix=''):
        for module_name, module in current_module.named_children():
            full_name = f"{prefix}.{module_name}" if prefix else module_name
            if module.__class__ in self.wrap_rule:
                if full_name in self.exceptions:
                    new_module = self.exceptions[full_name].wrap_module(module)
                    setattr(current_module, module_name, new_module)
                else:
                    new_module = self.wrap_rule[module.__class__].wrap_module(module)
                    setattr(current_module, module_name, new_module)
            else:
                self.wrap_model(module, full_name)
