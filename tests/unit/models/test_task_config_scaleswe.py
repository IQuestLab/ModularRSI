import pytest
from pydantic import ValidationError

from harbor.models.task.config import TaskConfig


SCALESWE_TOML = """
startup_required = true
agent_task_upload_allowed = false
fresh_environment_required = true
verifier_workspace_handoff_path = "/workspace/repo"
verifier_workspace_handoff_excludes = [".venv", "*/.venv"]

[verifier]
environment_mode = "separate"

[environment]
docker_image = "example/scaleswe:task"
network_mode = "no-network"
docker_image_expected_ids = ["sha256:image"]
docker_image_expected_rootfs_diff_ids = ["sha256:layer"]

[verifier.environment]
docker_image = "example/scaleswe:task"
network_mode = "no-network"
memory = "4G"
"""


def test_scaleswe_lifecycle_fields_are_preserved() -> None:
    config = TaskConfig.model_validate_toml(SCALESWE_TOML)

    assert config.startup_required is True
    assert config.agent_task_upload_allowed is False
    assert config.fresh_environment_required is True
    assert config.environment.allow_internet is False
    assert config.environment.docker_image_expected_ids == ["sha256:image"]
    assert config.verifier.environment_mode == "separate"
    assert config.verifier.environment is not None
    assert config.verifier.environment["memory"] == "4G"


def test_fresh_verifier_contract_rejects_shared_mode() -> None:
    with pytest.raises(ValidationError, match="fresh_environment_required"):
        TaskConfig.model_validate(
            {
                "fresh_environment_required": True,
                "verifier": {"environment_mode": "shared"},
            }
        )


def test_verifier_environment_implies_separate_mode() -> None:
    config = TaskConfig.model_validate(
        {"verifier": {"environment": {"docker_image": "example/image:tag"}}}
    )

    assert config.verifier.environment_mode == "separate"
