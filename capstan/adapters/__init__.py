import importlib.abc
import importlib.util
import sys

_WRAPPED = "_capstan_wrapped"


def wrap(owner, name, make):
    original = getattr(owner, name)
    if getattr(original, _WRAPPED, False):
        return
    wrapper = make(original)
    setattr(wrapper, _WRAPPED, True)
    setattr(owner, name, wrapper)


# Patch a module once something else imports it, so a hook never imports a framework into a process that would not.
def when_imported(name, patch):
    if name in sys.modules:
        patch(sys.modules[name])
    else:
        sys.meta_path.insert(0, _PatchOnImport(name, patch))


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, name, patch):
        self.name = name
        self.patch = patch

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            self.patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec
