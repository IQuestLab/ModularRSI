from __future__ import annotations

from pathlib import Path, PurePosixPath

from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.optional_import import MissingExtraError

try:
    from dirhash import dirhash
    from dockerfile_parse import DockerfileParser
    from e2b import AsyncSandbox, AsyncTemplate, FileType, Template
    from e2b.sandbox.commands.command_handle import CommandExitException
    from e2b.sandbox.filesystem.filesystem import WriteEntry

    _HAS_E2B = True
except ImportError:
    _HAS_E2B = False


class E2BEnvironment(BaseEnvironment):
    _UPLOAD_BATCH_SIZE = 20
    # e2b rejects sandbox creation with "Timeout cannot be greater than 1 hours"
    # (max 3600s). Upstream hardcoded 86_400 (24h) here, which 400s every trial.
    # Cap at the e2b maximum so tb2 trials can actually start.
    _SANDBOX_TIMEOUT_SEC = 3600

    @classmethod
    def preflight(cls) -> None:
        import os

        if not os.environ.get("E2B_API_KEY"):
            raise SystemExit(
                "E2B requires E2B_API_KEY to be set. "
                "Please set this environment variable and try again."
            )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        **kwargs,
    ):
        if not _HAS_E2B:
            raise MissingExtraError(package="e2b", extra="e2b")

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._workdir = next(
            (
                instruction["value"]
                for instruction in reversed(
                    DockerfileParser(
                        path=str(self._environment_definition_path)
                    ).structure
                )
                if instruction.get("instruction") == "WORKDIR"
            ),
            None,
        )

        self._sandbox: AsyncSandbox | None = None
        self._template_name = f"{environment_name}__{dirhash(self.environment_dir, 'sha256')[:8]}".replace(
            "/", "__"
        ).replace(".", "-")

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.E2B

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        if not self._environment_definition_path.exists():
            raise FileNotFoundError(
                f"{self._environment_definition_path} not found. Please ensure the "
                "file exists."
            )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_template(self):
        if self.task_env_config.docker_image:
            template = Template().from_image(
                image=self.task_env_config.docker_image,
            )
        else:
            template = Template(
                file_context_path=str(self.environment_dir),
            ).from_dockerfile(
                dockerfile_content_or_path=str(self._environment_definition_path),
            )

        await AsyncTemplate.build(
            template=template,
            alias=self._template_name,
            cpu_count=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(self):
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
        }

        self._sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata=metadata,
            timeout=self._SANDBOX_TIMEOUT_SEC,
            allow_internet_access=self.task_env_config.allow_internet,
        )

    async def _does_template_exist(self) -> bool:
        return await AsyncTemplate.alias_exists(self._template_name)

    def _template_lock_path(self) -> Path:
        """跨进程锁文件:同一 (账号, 模板) 只允许一个进程构建。

        模板名 = 环境名 + 环境目录 hash,与"哪个配置在跑"无关,所以同一道题在多个
        并发 harbor run 里解析出【同一个 alias】。而模板命名空间是按 e2b team 分的,
        不同账号各建各的、互不影响 —— 所以锁的粒度必须是 (账号, 模板名),
        用 API key 的 hash 代表账号(不落明文 key)。
        """
        import hashlib
        import os
        import tempfile

        acct = hashlib.sha256(
            os.environ.get("E2B_API_KEY", "").encode()
        ).hexdigest()[:8]
        lock_dir = Path(tempfile.gettempdir()) / "harbor_e2b_tpl_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        return lock_dir / f"{acct}_{self._template_name}.lock"

    async def _ensure_template(self, force_build: bool) -> None:
        """在跨进程互斥下完成"检查是否存在 → 构建"。

        为什么需要锁(2026-08-05 实测):`start()` 原本是 check-then-build,两步之间
        没有任何互斥。六个配置并行评同一道题时,六个进程同时判定 alias 不存在,于是
        一起构建同一个 alias,互相踩成:
            BuildException: build was cancelled
            SandboxException: 404: tag 'default' does not exist for template '...'
        单进程跑同一道题则完全正常。加锁后,第一个进程构建、其余等待并直接复用。
        flock 是阻塞调用,丢到线程里做,别卡住事件循环。
        """
        import asyncio
        import fcntl

        lock_path = self._template_lock_path()

        def _acquire():
            fh = open(lock_path, "w")
            fcntl.flock(fh, fcntl.LOCK_EX)
            return fh

        fh = await asyncio.to_thread(_acquire)
        try:
            # 拿到锁后【重新】检查:等锁期间别的进程很可能已经建好了
            if force_build or not await self._does_template_exist():
                self.logger.debug(f"Creating template {self._template_name}")
                await self._create_template()
        finally:
            await asyncio.to_thread(
                lambda: (fcntl.flock(fh, fcntl.LOCK_UN), fh.close())
            )

    @staticmethod
    def _is_missing_build_error(exc: BaseException) -> bool:
        """alias 存在、但没有可用 build 的那种 404。

        长这样:404: tag 'default' does not exist for template '<team>/<alias>'
        """
        msg = str(exc)
        return "does not exist for template" in msg or (
            "404" in msg and "template" in msg
        )

    async def start(self, force_build: bool):
        await self._ensure_template(force_build)

        try:
            await self._create_sandbox()
        except Exception as exc:
            # 自愈:`alias_exists()` 只看 alias 在不在,不看它有没有构建成功的 build。
            # 2026-08-05 的并发构建竞态留下了一批"空壳 alias"——存在,但没有 default
            # tag。碰上这种,_ensure_template 会以为已就绪而跳过构建,于是这里 404。
            # 强制重建一次即可修好,并且修好后同一 alias 对所有后续 trial 都有效。
            if not self._is_missing_build_error(exc):
                raise
            self.logger.warning(
                f"Template {self._template_name} has no usable build "
                f"(stale alias); forcing a rebuild."
            )
            await self._ensure_template(force_build=True)
            await self._create_sandbox()

        if not self._sandbox:
            raise RuntimeError(
                "Sandbox not found but was just created. This should never happen."
            )

        await self._sandbox.files.make_dir(str(EnvironmentPaths.agent_dir))
        await self._sandbox.files.make_dir(str(EnvironmentPaths.verifier_dir))

        # Make log directories world-writable so non-root agent/verifier
        # users can write to them.
        await self.exec(
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _stop_sandbox(self):
        if self._sandbox:
            await self._sandbox.kill()  # type: ignore[call-arg]

    async def stop(self, delete: bool):
        """Stops the environment and optionally deletes it."""
        if not delete:
            self.logger.info(
                "E2B harbor are ephemeral and will be deleted after use, "
                "regardless of delete=False."
            )

        if self._sandbox:
            try:
                await self._stop_sandbox()
            except Exception as e:
                self.logger.error(f"Error stopping sandbox: {e}")
            finally:
                self._sandbox = None
        else:
            self.logger.info("Sandbox has already been removed.")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def upload_file(self, source_path: Path | str, target_path: str):
        """
        Adds a local file to the environment.

        Args:
            source_path: The path to the source local file.
            target_path: The path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        await self._sandbox.files.write(target_path, Path(source_path).read_bytes())

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        """
        Adds a local directory to the environment.

        Args:
            source_dir: The path to the source local directory.
            target_dir: The path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        files: list[WriteEntry] = []
        for file_path in Path(source_dir).rglob("*"):
            if file_path.is_file():
                files.append(
                    WriteEntry(
                        path=str(
                            PurePosixPath(target_dir)
                            / file_path.relative_to(Path(source_dir)).as_posix()
                        ),
                        data=file_path.read_bytes(),
                    )
                )

        if files:
            for i in range(0, len(files), self._UPLOAD_BATCH_SIZE):
                batch = files[i : i + self._UPLOAD_BATCH_SIZE]
                await self._sandbox.files.write_files(batch)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def download_file(self, source_path: str, target_path: Path | str):
        """
        Downloads a file from the environment to the local machine.

        Args:
            source_path: The path to the source file in the environment.
            target_path: The local path to which to copy the file.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        Path(target_path).write_bytes(
            await self._sandbox.files.read(source_path, format="bytes"),
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def download_dir(self, source_dir: str, target_dir: Path | str):
        """
        Downloads a directory from the environment to the local machine. This overwrites
        existing files in the target directory.

        Args:
            source_dir: The path to the source directory in the environment.
            target_dir: The local path to which to copy the directory.
        """
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        results = await self._sandbox.files.list(source_dir)

        for result in results:
            if result.type == FileType.DIR:
                sub_target_dir = Path(target_dir) / Path(result.path).relative_to(
                    Path(source_dir)
                )
                sub_target_dir.mkdir(parents=True, exist_ok=True)

                await self.download_dir(
                    source_dir=result.path,
                    target_dir=sub_target_dir,
                )

            if result.type == FileType.FILE:
                target_path = Path(target_dir) / Path(result.path).relative_to(
                    Path(source_dir)
                )

                target_path.parent.mkdir(parents=True, exist_ok=True)

                await self.download_file(
                    source_path=result.path,
                    target_path=str(target_path),
                )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        info = await self._sandbox.files.get_info(path)
        return info.type == FileType.DIR

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        info = await self._sandbox.files.get_info(path)
        return info.type == FileType.FILE

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """
        Executes a command in the environment.

        Args:
            command: The command to execute.
            cwd: The working directory in which to execute the command.
            env: The environment  variables to set.
            timeout_sec: The timeout in seconds.
            user: Username or UID to run the command as. None falls back to
                ``self.default_user``; if that is also None the sandbox default is used.
        """
        user = self._resolve_user(user)
        env = self._merge_env(env)

        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")

        handle = await self._sandbox.commands.run(
            cmd=command,
            background=True,
            cwd=cwd or self.task_env_config.workdir or self._workdir,
            envs=env,
            timeout=timeout_sec or 0,
            user=str(user) if user is not None else "root",
        )

        try:
            result = await handle.wait()
        except CommandExitException as e:
            result = e

        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.exit_code,
        )
