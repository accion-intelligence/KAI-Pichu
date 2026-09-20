"""The packaged task-authoring skill and its installer."""
from pathlib import Path

import pytest

from kai_pichu.cli import main
from kai_pichu.skills import SKILL_NAME, install_skill, skill_destination, skill_source


def front_matter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    header = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in header.strip().splitlines())


def test_skill_has_name_and_description_front_matter():
    text = (skill_source() / "SKILL.md").read_text(encoding="utf-8")
    header = front_matter(text)
    assert header["name"] == SKILL_NAME
    assert "kind: fusion" in header["description"]
    assert "kai-pichu benchmark guide" in text


def test_skill_installs_for_both_agents_under_the_project_root(tmp_path, capsys):
    assert main(["skill", "install", "--directory", str(tmp_path)]) == 0
    for agent, folder in (("claude", ".claude"), ("codex", ".codex")):
        installed = tmp_path / folder / "skills" / SKILL_NAME / "SKILL.md"
        assert installed.read_text() == (skill_source() / "SKILL.md").read_text()
        assert skill_destination(agent, scope="project", root=tmp_path) == installed.parent
    assert "Installed codex skill" in capsys.readouterr().out


def test_skill_install_refuses_to_overwrite_without_force(tmp_path):
    install_skill("claude", scope="project", root=tmp_path)
    with pytest.raises(ValueError, match="--force"):
        install_skill("claude", scope="project", root=tmp_path)
    marker = tmp_path / ".claude" / "skills" / SKILL_NAME / "stale.txt"
    marker.write_text("old")
    install_skill("claude", scope="project", root=tmp_path, force=True)
    assert not marker.exists()


def test_skill_user_scope_targets_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert skill_destination("codex", scope="user") == tmp_path / ".codex" / "skills" / SKILL_NAME
    with pytest.raises(ValueError, match="unknown agent"):
        skill_destination("cursor", scope="user")
