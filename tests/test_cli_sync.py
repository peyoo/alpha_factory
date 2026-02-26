from typer.testing import CliRunner
from unittest.mock import patch

from alpha_factory.cli.main import app


def test_sync_with_short_flags_interval():
    runner = CliRunner()
    with patch("alpha_factory.cli.data.TushareDataService") as MockService:
        instance = MockService.return_value
        instance.sync_data.return_value = None
        result = runner.invoke(app, ["sync", "-s", "20240101", "-e", "20240105"])
        assert result.exit_code == 0
        assert "准备同步数据" in result.output or "数据同步完成" in result.output


def test_sync_incremental_no_token():
    runner = CliRunner()
    # Patch the get_tushare_token used by the data module
    with patch("alpha_factory.cli.data.get_tushare_token", return_value=""):
        result = runner.invoke(app, ["sync"])
        # Expect non-zero exit due to missing token
        assert result.exit_code != 0
        assert "未检测到 TUSHARE_TOKEN" in result.output
