"""processtrak entry points for CADMap."""
from pathlib import Path
from urllib.request import pathname2url


def getstepurlpath():
    return [pathname2url(str(Path(__file__).parent))]
