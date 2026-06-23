"""
Runtime compatibility shims for this repo-local training environment.
"""

try:
    import importlib.metadata as _stdlib_metadata
except Exception:  # pragma: no cover
    _stdlib_metadata = None

try:
    import importlib_metadata as _backport_metadata
except Exception:  # pragma: no cover
    _backport_metadata = None


def _patch_version_api(module):
    if module is None:
        return

    original_version = module.version
    package_not_found = module.PackageNotFoundError

    def patched_version(name):
        if name == "torchao":
            raise package_not_found(name)
        return original_version(name)

    module.version = patched_version


_patch_version_api(_stdlib_metadata)
_patch_version_api(_backport_metadata)
