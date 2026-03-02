from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from alpha_factory.cli.main import app


def test_eval_uses_default_start_date_when_omitted():
    runner = CliRunner()

    with (
        patch("alpha_factory.cli.eval.DataProvider") as mock_dp_cls,
        patch("alpha_factory.cli.eval.single_factor_alpha_analysis") as mock_analysis,
    ):
        mock_lf = MagicMock()
        mock_dp_cls.return_value.load_pool_data.return_value = mock_lf
        mock_analysis.return_value = {"nav": [1.0, 1.01, 1.02]}

        result = runner.invoke(
            app,
            [
                "eval",
                "--expr",
                "CLOSE",
                "--no-report",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_dp_cls.return_value.load_pool_data.assert_called_once()
    _, start_date_arg, _ = mock_dp_cls.return_value.load_pool_data.call_args.args
    assert start_date_arg == "20190101"
