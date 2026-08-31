"""Load-time guards on the separate-verifier contract (ScaleSWE).

Both checks exist because the failure they prevent is expensive and quiet:

- an invalid ``[verifier].environment`` used to surface only when the trial
  reached grading, i.e. after the agent rollout was already spent;
- ``verifier_workspace_handoff_path`` is ``rm -rf``'d in the fresh verifier
  before the agent workspace is unpacked over it, so a system root there wipes
  the image's own tooling and yields a wrong grade rather than an error.
"""

import pytest
from pydantic import ValidationError

from harbor.models.task.config import TaskConfig


def _cfg(top_level: str = "", verifier_extra: str = "") -> TaskConfig:
    """Top-level keys MUST precede the [verifier] table — anything written after
    it lands inside that table instead (plain TOML semantics, and an easy way to
    write a test that silently asserts nothing)."""
    return TaskConfig.model_validate_toml(
        'schema_version = "1.0"\n'
        f"{top_level}"
        '\n[verifier]\nenvironment_mode = "separate"\n'
        f"{verifier_extra}"
    )


@pytest.mark.unit
class TestVerifierEnvironmentValidatedEagerly:
    def test_valid_verifier_environment_loads(self):
        cfg = _cfg(
            verifier_extra='\n[verifier.environment]\ndocker_image = "x:1"\ncpus = 2\n'
        )
        assert cfg.verifier.environment == {"docker_image": "x:1", "cpus": 2}

    def test_bad_verifier_environment_fails_at_load(self):
        with pytest.raises(ValidationError):
            _cfg(verifier_extra="\n[verifier.environment]\ncpus = 'not-a-number'\n")

    def test_unsupported_network_mode_in_verifier_env_fails_at_load(self):
        """The nested env goes through the same EnvironmentConfig validator, so
        its own rules (e.g. network_mode) apply here too."""
        with pytest.raises(ValidationError):
            _cfg(
                verifier_extra='\n[verifier.environment]\nnetwork_mode = "half-open"\n'
            )


@pytest.mark.unit
class TestHandoffPathGuard:
    def test_workspace_path_accepted(self):
        cfg = _cfg('verifier_workspace_handoff_path = "/workspace/repo"\n')
        assert cfg.verifier_workspace_handoff_path == "/workspace/repo"

    def test_relative_path_rejected(self):
        with pytest.raises(ValidationError, match="absolute"):
            _cfg('verifier_workspace_handoff_path = "repo"\n')

    @pytest.mark.parametrize("bad", ["/", "//", "/usr", "/etc/", "/root"])
    def test_system_roots_rejected(self, bad):
        """`/` and `/usr` pass a bare startswith('/') check — that was the hole."""
        with pytest.raises(ValidationError, match="system root"):
            _cfg(f'verifier_workspace_handoff_path = "{bad}"\n')

    def test_handoff_requires_separate_mode(self):
        with pytest.raises(ValidationError, match="separate verifier"):
            TaskConfig.model_validate_toml(
                'schema_version = "1.0"\n'
                'verifier_workspace_handoff_path = "/workspace/repo"\n'
            )
