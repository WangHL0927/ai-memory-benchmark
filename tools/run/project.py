"""Read the v0.7 project's primary label from the shared frozen transcript."""
from tools.trial import transcript


def declared_project(content):
    header, _ = transcript(content)
    return header['project（标注者填写）'].split(',', 1)[0].strip()
