"""The files the clones wrote, managed apart from any one conversation (#1554).

`artifacts.library.ArtifactLibrary` lists, opens, archives, restores and deletes the files
in the workspace's artifact folders, and opens a story into a conversation. It is the Core
service behind the Files screen; the HTTP routes in `ui/artifacts.py` only translate.
"""
