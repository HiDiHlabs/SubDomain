from importlib.metadata import PackageNotFoundError, version

from ._domaindetection import SubDomain

try:
    __version__ = version("subdomain")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = ["SubDomain"]
