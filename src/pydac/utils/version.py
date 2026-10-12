"""Single package version source; release versions are assigned at qualification."""

__version__ = "3.0.0"

# Set only by the qualified public-release build, never from runtime Git HEAD.
# When known: {"revision": <full SHA>, "url": <public GitHub commit URL>}.
PUBLIC_SOURCE = None
